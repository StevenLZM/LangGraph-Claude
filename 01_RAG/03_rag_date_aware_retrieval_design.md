# 工程设计：RAG 日期感知语义检索

## 1. 目标

在现有 `BM25 + 语义` 混合检索 + `parent/child` 两层架构的基础上，**叠加**日期感知能力，让系统能正确处理以下带时间意图的自然语言查询：

- 软意图：「最新的报销票据」「最近的合同」
- 硬意图-年份：「2024 年的报销」
- 硬意图-端点：「2023 年之前的票据」「2024 年之后的合同」
- 硬意图-区间：「2024 Q1 的报销」「上个月的发票」
- 无意图：「报销流程是什么」（走原流程，零回归）

本次设计不覆盖：时间类型细分（开票/消费/提交日期的语义区分）、时间衰减加权公式的精调、OCR 日期修复。

## 2. 核心约束

1. **保留 BM25 + 语义混合检索**：`EnsembleRetriever` 不动结构。
2. **保留 parent/child 两层架构**：child 检索 → parent hydrate 不动。
3. 时间感知作为**正交增强层**嵌入，不替换不重写现有链路。

## 3. 两个日期维度

| 字段 | 来源 | 含义 | ChromaDB 存储 |
|------|------|------|---------------|
| `upload_date` | 入库时 `time.strftime("%Y%m%d")` | 入库时刻 | 单值 int |
| `doc_date_min` / `doc_date_max` | 从内容抽取，多日期展平 | 业务日期区间（开票/消费/签订） | 两个标量 int |
| `has_doc_date` | 是否成功抽取到日期 | 二值过滤辅助 | bool |

说明：ChromaDB metadata 不支持数组，故多日期必须展平为 `min/max` 两个标量。无日期的文档 `min=max=0`、`has_doc_date=False`，有日期要求的查询用 `has_doc_date=True` 排除它们。

## 4. 三种检索策略对比

时间过滤 / 排序的插入点决定召回质量与召回完整性的权衡。

### 4.1 方案A：Pre-filter（先时间过滤，再混合检索）

```
Query → 时间意图解析 → 语义路：ChromaDB where filter 召回 K
                       BM25 路：召回 K·M 后 post-filter
                       ↓
                       Ensemble RRF 融合 → parent hydrate
```

- **优点**：候选集干净，TopK 全部满足时间约束；语义 / BM25 排序不被无关时段污染。
- **缺点**：BM25 不支持原生 metadata filter，需召回扩大化后再过滤；时间意图解析错了直接漏召，无兜底。
- **适用**：硬意图（year / before / after / range）。

### 4.2 方案B：Post-filter + Rerank（先混合检索，再时间精排）

```
Query → 时间意图解析 → Ensemble 召回 K·N（无过滤）
                       ↓
                       parent hydrate → 按时间二级排序 / 软过滤 → 返回 TopK
```

- **优点**：完全不破坏原召回链路；语义召回完整性最高；对软意图天然友好。
- **缺点**：召回扩大化有计算浪费；硬意图下 TopK 仍可能被无关时段的高相似度文档挤占。
- **适用**：软意图（latest）；无意图查询的兜底。

### 4.3 方案C：双轨融合（语义路 + 时间路并行）

```
Query → 一路：Ensemble（无时间约束）召回 K
        二路：纯 metadata 时间过滤 + 日期 desc 召回 K
        → RRF 融合 child 结果 → parent hydrate
```

- **优点**：相关性与时效性双兜底。
- **缺点**：架构复杂度提升一档，融合权重难调；两路重叠少时 RRF 效果不稳定。
- **适用**：暂不采用，作为后续优化储备。

### 4.4 最终选型：按意图类型分派（A + B 混合）

| 意图类型 | 例子 | 策略 | 理由 |
|----------|------|------|------|
| `year` / `before` / `after` / `range` | "2024年"、"Q1"、"2023年之前" | **方案A** | 用户表达确定，硬过滤代价可接受 |
| `latest` | "最新的"、"最近的" | **方案B** | 软意图，作为排序信号更合理 |
| `none` | "报销流程是什么" | **原流程** | 不引入时间逻辑，零回归 |

这样既不破坏 ensemble 结构（A 仅在两个 retriever 内部各自加 filter，融合方式不变；B 仅在 hydrate 后加排序），也不引入新的检索器实例。

## 5. 改动清单

### 5.1 新增 `rag/date_extractor.py`

输入 parent chunk 文本，输出 `{min: int, max: int, found: bool}`（YYYYMMDD）。

- **正则兜底**：匹配 `YYYY年MM月DD日` / `YYYY-MM-DD` / `YYYY/MM/DD` / `YYYY.MM.DD` / `YYYY年MM月`。
- **LLM 回退**：正则未命中时调 `_get_llm()`（DashScope Qwen），低温度，约束 JSON 输出。
- **缓存**：`data/date_cache.sqlite`，key = `doc_id + chunk_hash`。
- 多日期取 `min/max` 转 YYYYMMDD int；无日期写 `0`。

### 5.2 改造 `rag/chunker.py:137-154`

日期抽取在 parent 循环内、child 循环外执行（parent 级），child 继承。新增字段：

```python
"upload_date": int(time.strftime("%Y%m%d")),  # 替换原 upload_time
"doc_date_min": dates.min,
"doc_date_max": dates.max,
"has_doc_date": dates.found,
```

### 5.3 新增 `rag/query_rewriter.py`

从 `chain.py` 抽出原 rewrite 逻辑，并扩展为结构化 JSON 输出：

```json
{
  "rewritten_query": "报销票据",
  "time_intent": {
    "type": "latest | year | before | after | range | none",
    "field": "doc_date | upload_date",
    "range": { "gte": 20240101, "lte": 20241231 } | null,
    "sort": "desc | null"
  }
}
```

六类意图判定规则（以今日为锚点）：

- 含「最新提交/上传/归档」→ `field=upload_date`，否则默认 `doc_date`。
- 「最新/最近」+ 无具体时间 → `type=latest`，`range=null`，`sort=desc`。
- 「YYYY 年」→ `type=year`，`range={gte: YYYY0101, lte: YYYY1231}`。
- 「YYYY 年之前/以前」→ `type=before`，`range={gte: 0, lte: (YYYY-1)1231}`。
- 「YYYY 年之后/以来」→ `type=after`，`range={gte: (YYYY+1)0101, lte: 99991231}`。
- 「Q1/Q2/Q3/Q4」「上个月」「近 N 天」→ `type=range`，按锚点算具体区间。
- 无时间词 → `type=none`。

### 5.4 扩展 `rag/vectorstore.py:208-234`

`similarity_search_with_threshold()` 新增 `metadata_filter: dict | None`，转 ChromaDB `where` 语法：

```python
where = {"$and": [
    {"has_doc_date": True},
    {f"{field}_max": {"$gte": gte}},
    {f"{field}_min": {"$lte": lte}},
]}
```

`upload_date` 单值字段直接用 `{"upload_date": {"$gte": gte, "$lte": lte}}`。

### 5.5 改造 `rag/retriever.py`

不改 `EnsembleRetriever` 本身，把 filter 注入两个子 retriever：

- **语义路**：ChromaDB native `filter`，`k = SEMANTIC_TOP_K * 2`（防过滤后不足）。
- **BM25 路**：新增 `TimeFilteredBM25Wrapper` 包装，`invoke()` 后按 metadata post-filter，截断到原 `k`。

`hydrate_parent_results()` 签名加 `time_intent`。排序 key：

| `time_intent.type` | 排序 key |
|--------------------|---------|
| `latest` | `(-date, -score, -child_count)` |
| `year / range / before / after` | `(-score, -child_count, -date)` |
| `none` | 原 key 不变 |

注意：现 `get_hybrid_retriever()` 是单例。需缓存 base ensemble，按 query 动态包装 filter，避免 BM25 索引重建开销。

### 5.6 `rag/chain.py` 接线

```python
rewrite_result = query_rewriter.invoke(query)  # 返回 dict
docs = retrieve_with_hybrid(
    rewrite_result["rewritten_query"],
    time_intent=rewrite_result["time_intent"],
)
```

对外 `answer(query)` 签名不变。

### 5.7 `config.py:72-97` 新增

```python
DATE_EXTRACTION_ENABLED: bool = True
DATE_EXTRACTION_LLM_FALLBACK: bool = True
DATE_CACHE_PATH: str = "data/date_cache.sqlite"
HARD_FILTER_K_MULTIPLIER: int = 2
```

## 6. 关键文件一览

| 文件 | 操作 | 关键位置 |
|------|------|----------|
| `rag/date_extractor.py` | 新建 | — |
| `rag/query_rewriter.py` | 新建 | 从 chain.py 抽出 + 扩展时间意图 |
| `rag/chunker.py` | 修改 | L137-154 child metadata；parent 循环内补抽取 |
| `rag/vectorstore.py` | 修改 | L208-234 增 `metadata_filter` |
| `rag/retriever.py` | 修改 | L35-62 注入 filter；L83-172 加二级排序 key |
| `rag/chain.py` | 修改 | L128-134 改用 query_rewriter 模块 |
| `config.py` | 修改 | L72-97 新增配置 |

## 7. 复用的现有能力

- `_get_llm()` / `_get_rewrite_llm()`（`chain.py:20-56`）— 日期抽取器与 rewriter 复用。
- `vectorstore.similarity_search_with_threshold()` 的 `filter_doc_ids` 参数雏形 — 扩展而非重写。
- `EnsembleRetriever` + RRF 融合机制 — 完全不动。
- `hydrate_parent_results()` 聚合 / 排序框架 — 仅替换排序 key。

## 8. 验证方案

### 8.1 单元验证

- `date_extractor`：10 份样本（含/不含/多日期/中英/混排），断言 `min/max/found`。
- `query_rewriter`：六类 query 各 2 个样例，断言 `time_intent.type/field/range`。

### 8.2 端到端验证（按意图分组）

1. **none**：`answer("报销流程是什么")` — 对比 baseline 召回结果应一致。
2. **latest（方案B）**：`answer("最新的报销票据")` — 日志无 metadata filter，hydrate 排序 key 为 `(-doc_date_max, ...)`。
3. **year（方案A）**：`answer("2024 年的报销")` — 日志有 ChromaDB where filter，TopK 全部满足区间约束。
4. **before / after / range**：类似 year，验证区间边界。
5. **upload field**：`answer("最近上传的合同")` — `field=upload_date`。

### 8.3 回归与成本检查

- 关 `DATE_EXTRACTION_ENABLED`，全部 query 退化为原行为，确认无回归。
- ingest 阶段统计 LLM 抽取调用次数 + 缓存命中率（第二次 ingest 同文档应 100% 命中）。
- 检索耗时对比：方案A（硬过滤）应 ≤ baseline；方案B（召回扩大化）增长 ≤ 30%。
- BM25 wrapper post-filter 召回率：过滤后剩余 / 原 K ≥ 0.8。

### 8.4 离线评价接入

日期感知检索必须进入 `evals/` 评测体系，而不是只看日志：

- `reference`：用自然语言写清“应该按什么日期规则选择什么答案”，供 RAGAS 对比最终回答。
- `expected_sources`：保留为人工排查字段，帮助定位时间类问题命中的来源文档。
- `category=time`：时间类样本单独分组，避免被普通概念题平均分掩盖。

推荐命令：

```bash
python -m evals.run --dry-run
python -m evals.run
```

日期检索上线门槛建议：

- 时间类样本 `context_recall` 不低于普通样本 5 个百分点以上
- 时间类样本 `answer_correctness` 不低于 0.85
- 若 `context_recall` 高但 `answer_correctness` 低，优先排查 prompt 和日期比较逻辑

RAGAS 评估体系的整体设计见 `05_rag_ragas_evaluation_design.md`。本设计文档只定义时间类样本如何进入该评估体系。

## 9. 实施顺序

1. `query_rewriter.py`（可独立验证，不依赖索引重建）
2. `date_extractor.py` + 单元测试
3. `chunker.py` 接入日期抽取 → 重新 ingest
4. `vectorstore.py` 增 metadata_filter
5. `retriever.py` 注入 filter + 二级排序
6. `chain.py` 接线
7. 端到端验证
