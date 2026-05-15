# 工程设计：结构化 Section 切分 + Atomic 块保护

## 1. 背景与问题

`Chunking V2` 已经把单层 chunk 升级为 parent-child 双层，但 section 边界仍依赖 `_is_heading` 的正则启发式：

```python
# rag/chunker.py（升级前）
heading_patterns = [
    r"^第[一二三四五六七八九十百千\d]+[章节篇部分]",
    r"^[0-9]+[.)、．]",
    r"^[一二三四五六七八九十]+、",
    r"^[A-Z][A-Za-z0-9 _-]{1,40}$",
]
```

由此带来两类典型缺陷：

| 缺陷 | 例子 | 后果 |
|---|---|---|
| Heading 漏识别 | "3.2.1 检索流程"、Markdown `## xxx`、加粗短行 | section 错误合并，`section_path` 失真 |
| 表格被腰斩 | `\| 列A \| 列B \|` 多行表格 | child splitter 按字符切，markdown 表格断裂，召回后 LLM 看不懂 |

本次升级在 **不改变 parent-child 索引结构** 的前提下，把 section 边界识别从"猜"升级为"读 PDF 结构"，并新增 atomic 块保护机制。

## 2. 核心改动概览

| 模块 | 文件 | 关键函数 |
|---|---|---|
| Loader 结构化抽取 | `rag/loader.py` | `_extract_structured_blocks` / `_rows_to_markdown` |
| Loader 分页合并 | `rag/loader.py` | `_merge_broken_sentences`（升级 3 元组） |
| Chunker 结构化路径 | `rag/chunker.py` | `_has_structured` / `_build_sections_structured` |
| Chunker 原子块保护 | `rag/chunker.py` | `_protect_atomics` / `_restore_atomics` |
| Chunker 主流程路由 | `rag/chunker.py` | `chunk_documents` |

## 3. Loader：从字号读 heading，从 find_tables 读表格

### 3.1 输出契约

`load_pdf_pymupdf` 现在在每个 page Document 的 metadata 里追加：

```python
metadata["structured_blocks"] = [
    {"type": "heading", "level": 1, "text": "第一章 概述", "page": 1},
    {"type": "paragraph", "text": "本系统支持检索增强生成。", "page": 1},
    {"type": "table",     "text": "| 列A | 列B |\n| --- | --- |\n| 1 | 2 |", "page": 1},
    ...
]
```

block 顺序按页面 y 坐标排序，等同于阅读顺序。

### 3.2 Heading 识别规则

`_extract_structured_blocks` 的判定：

1. 收集页面所有 span 的 `size` → 取中位数作为 **正文字号 body_size**
2. 单个 block 满足下面任一条件 → heading：
   - `max_size >= body_size * 1.18`
   - `bold and len(text) <= 40 and "\n" not in text`
3. 按字号比例分级：
   - `ratio >= 1.6` → level 1
   - `ratio >= 1.3` → level 2
   - 其余 → level 3

为什么用中位数而不是均值：长文档里正文 span 数量远大于 heading，中位数天然反映"主体字号"，不被少量大字号 heading 拉偏。

### 3.3 表格识别规则

```python
tables = page.find_tables()  # pymupdf >=1.23
for tbl in tables:
    rows = tbl.extract()
    md = _rows_to_markdown(rows)
```

- 表格区域的 bbox 单独存到 `table_rects`
- 后续遍历文本 block 时，**若 block bbox 落在任一 table_rect 内，跳过** —— 防止表格内文字被同时当作 paragraph 重复抽取
- `_rows_to_markdown` 做单元格去 `\n`、转义 `|`、补齐列数

### 3.4 跨页合并的连带升级

`_merge_broken_sentences` 升级签名：

```python
List[Tuple[int, str]]  →  List[Tuple[int, str, List[dict]]]
```

合并跨页"断句"时，同步把 `last_blocks + blocks` 拼起来，避免 page 2 被合并进 page 1 后 page 2 的 structured_blocks 丢失。

## 4. Chunker：路由 + Atomic 保护

### 4.1 路由

```python
if _has_structured(doc_group):
    sections = _build_sections_structured(doc_group)
else:
    sections = _build_sections(doc_group)   # 旧的正则路径，作为 fallback
```

`_has_structured` 检查任一 doc 是否提供了 `structured_blocks`。pypdf 兜底路径仍然走旧逻辑，保持向后兼容（`test_chunk_documents_returns_hierarchical_result` 即走该路径）。

### 4.2 结构化 section 构建

`_build_sections_structured` 的状态机：

```
heading 出现 → flush 当前 section → section_path = heading text，level = blk.level
paragraph    → 累入 current_parts (kind="text")
table/code   → 累入 current_parts (kind="atomic")，并记入 current_atomics
```

`flush()` 时，对 `kind="text"` 的部分单独 `_normalize_text`，atomic 部分**原样拼接**，避免把 markdown 表格的换行折掉。

section.metadata 多出两个字段：
- `heading_level`：1/2/3，下游可用于层级检索
- `atomic_texts`：tuple[str, ...]，给后续 splitter 使用

### 4.3 Atomic 保护机制

核心思路：**在 splitter / normalize 看不到原始 atomic 文本，只看到占位符**。

```python
ATOMIC_PLACEHOLDER_PREFIX = "\x00ATOM"
ATOMIC_PLACEHOLDER_SUFFIX = "\x00"

def _protect_atomics(text, atomics):
    for i, atomic in enumerate(atomics):
        ph = f"\x00ATOM{i}\x00"
        text = text.replace(atomic, f"\n\n{ph}\n\n", 1)  # 首次出现替换
    return text, placeholder_map

def _restore_atomics(text, placeholder_map):
    for ph, atomic in placeholder_map.items():
        text = text.replace(ph, atomic)
    return text
```

设计要点：

| 决策 | 原因 |
|---|---|
| 用 `\x00` 作前后缀 | PDF 文本极少出现空字符，不会误碰 |
| 占位符前后包 `\n\n` | 让 `RecursiveCharacterTextSplitter` 在边界优先切分，整段 atomic 进入同一个 chunk |
| `_should_merge_lines` 跳过含占位符的行 | 防止 `_normalize_text` 的"无标点合并下一行"逻辑把占位符合并到正文里 |
| 占位符长度短 (<10 字符) | splitter 内部填充时不会因 chunk_size 不足而切断 |

### 4.4 Parent / Child 切分时序

升级后正确的时序是 **先在占位符态 normalize，再 restore**，否则表格换行会被吞：

```python
work_text, placeholder_map = _protect_atomics(section.text, atomics)
parent_works = parent_splitter.split_text(work_text)

for parent_work in parent_works:
    # ✅ 正确：normalize 时占位符还在原位
    normalized_parent = _restore_atomics(_normalize_text(parent_work), placeholder_map)

    child_works = child_splitter.split_text(parent_work)
    for child_work in child_works:
        is_pure_atomic = child_work.strip() in placeholder_map
        if is_pure_atomic:
            # 整个 child 就是一个表格 —— 不做 normalize，原样输出
            normalized_child = placeholder_map[child_work.strip()].strip()
        else:
            normalized_child = _restore_atomics(_normalize_text(child_work), placeholder_map)
```

### 4.5 child metadata 新增字段

```python
child_metadata = {
    ...
    "heading_level": heading_level,
    "is_atomic": is_pure_atomic,   # True 表示该 child 是完整表格/代码
    ...
}
```

`is_atomic=True` 的 child 在下游 reranker / 上下文格式化时可以特殊对待（例如不截断、按 markdown 表格高亮）。

## 5. 兼容性与回归

| 场景 | 行为 |
|---|---|
| pymupdf 加载成功且抽到 structured_blocks | 走新路径，section 边界来自字号 + heading 列表 |
| pymupdf find_tables 不可用 / 抛异常 | structured_blocks 仅含 heading + paragraph，链路不中断 |
| 加载 fallback 到 pypdf | metadata 不含 `structured_blocks`，自动回到旧的正则 section 切分 |
| 旧测试 `test_chunk_documents_returns_hierarchical_result` | 输入是手工构造 Document（无 structured_blocks），完全走旧路径 → 通过 |

## 6. 待办与已知边界

- 大表格（行数极多）会成为单个超大 child，超过 `CHILD_MAX_TOKENS=360`。当前选择不切分以保结构完整，但若发现 embedding 截断，需要为表格单独引入"按行打包"的二次切分
- `_protect_atomics` 用 `text.find(atomic)` 首次匹配 + `replace(..., 1)`，若同一 section 内出现多个完全相同的 atomic 文本，第二次起占位失败、回退为原文（不会出错，只是损失保护）
- heading level 仅按字号比例，不参考缩进/编号深度。Markdown 文档建议另走 `MarkdownHeaderTextSplitter` 路径
- 代码块（`type="code"`）目前 loader 不主动识别（PDF 内罕见），契约已预留，未来可由 markdown loader 注入

## 7. 验收用例

```python
docs = [Document(page_content="占位", metadata={
    "doc_id": "d1", "page": 1, "total_pages": 1, "source": "x.pdf", "file_path": "/tmp/x.pdf",
    "structured_blocks": [
        {"type": "heading",   "level": 1, "text": "第一章 概述", "page": 1},
        {"type": "paragraph", "text": "本系统支持检索增强生成。", "page": 1},
        {"type": "table",     "text": "| 列A | 列B |\n| --- | --- |\n| 1 | 2 |", "page": 1},
        {"type": "heading",   "level": 2, "text": "第二章 规格", "page": 2},
        {"type": "paragraph", "text": "Python 3.9+。", "page": 2},
    ],
})]

res = chunk_documents(docs)
# 预期：
#   2 个 section（第一章、第二章），各对应 1 个 parent
#   parent 内容保留 markdown 表格的换行
#   children 中表格作为完整 child 输出（is_atomic=True）
```

实测输出（节选）：

```
--P-- 第一章 概述 L 1
第一章 概述

本系统支持检索增强生成。

| 列A | 列B |
| --- | --- |
| 1 | 2 |
```

结构化切分上线后还需要跑离线评测，避免只验证"表格没被切坏"，却忽略检索排序质量：

```bash
python -m evals.run --dry-run
python -m evals.run
```

重点观察：

- `Context Completeness`：命中的 parent 是否包含回答所需关键词
- `Parent Hit Rate`：child 召回后是否能回填到正确 parent
- `nDCG@5 / MRR`：结构化 section 调整后相关 parent 是否排得更靠前
- 表格/代码类样本应单独放入 `evals/dataset.jsonl`，避免被普通文本样本掩盖

## 8. 与现有文档的关系

- `02_rag_chunking_v2_design.md`：定义 parent-child 结构与 ID 规则，本次升级**完全沿用**
- `03_rag_date_aware_retrieval_design.md`：日期抽取放在 parent 级一次调用，本次升级**未触动**
- 本文档（`04`）只在"section 边界识别"和"atomic 内容保护"两点上深化
