# 工程设计：RAG Chunking V2 升级方案

## 1. 目标

将当前“字符长度驱动的单层分块”升级为可用于生产演进的 `parent-child` 索引架构，核心目标：

- 分块从字符上限切换为 token 约束
- 索引从单层 chunk 切换为 parent / child 双层结构
- 检索从“直接返回子块”切换为“子块召回、父块回填”
- 元数据从 `chunk_index` 主导切换为稳定内容 ID 主导

本次实现不覆盖 OCR 修复、表格结构恢复、多租户隔离和在线双读双写。

## 2. 核心改动

### 2.1 分块层 `rag/chunker.py`

- 新增 `ChunkingResult`，统一返回：
  - `parents`
  - `children`
  - `stats`
- 分块流程改为三步：
  1. 文本轻量规范化
  2. 标题/空行驱动的 section 切分
  3. section 内按 token 约束生成 parent，再从 parent 切 child
- 默认参数：
  - `PARENT_TARGET_TOKENS=900`
  - `PARENT_MAX_TOKENS=1200`
  - `PARENT_OVERLAP_TOKENS=100`
  - `CHILD_TARGET_TOKENS=280`
  - `CHILD_MAX_TOKENS=360`
  - `CHILD_OVERLAP_TOKENS=60`
- 每个 parent / child 都携带：
  - `doc_id`
  - `doc_version`
  - `section_path`
  - `page_start/page_end/page_range`
  - `token_count`
- 稳定 ID 由内容哈希生成：
  - `parent_id = {doc_id}:p:{sha256(text)[:12]}`
  - `child_id = {doc_id}:c:{sha256(text)[:12]}`

### 2.2 存储层

#### Child vectorstore `rag/vectorstore.py`

- Chroma collection 名称改为版本化：
  - `rag_knowledge_base_v2_children`
- `add_documents()` 改为接收 `ChunkingResult`
- 仅 child chunks 写入向量库
- 写入失败时执行文档级回滚

#### Parent docstore `rag/docstore.py`

- 新增 SQLite docstore：
  - 路径：`data/docstore/parents_v2.sqlite`
- parent chunk 保存原文和完整 metadata
- 支持：
  - `upsert_parents`
  - `get_parents`
  - `delete_document`
  - `list_documents`

### 2.3 检索层 `rag/retriever.py`

- 保留 Dense + BM25 + Ensemble 召回，但召回对象改为 child
- 新增 `ParentChildHybridRetriever`
- child 召回后执行 parent 聚合：
  - 按 `parent_id` 去重
  - 聚合命中的 `child_id`
  - 按 child 排名分数排序
  - 从 docstore 回填 parent 原文
- 最终返回给上游链路的是 hydrated parent documents，而不是 child documents

### 2.4 链路与 UI

- `rag/chain.py`
  - `format_docs_for_context()` 支持 `page_range`
  - 上下文头部可展示 `section_path`
- `app.py`
  - 上传索引直接走 v2 分块
  - 侧边栏状态展示 parent / child 数量
  - 文档列表展示“父块 / 子块 / 页数”

## 3. 配置改动

`config.py` 新增：

- `TOKENIZER_NAME`
- `PARENT_TARGET_TOKENS`
- `PARENT_MAX_TOKENS`
- `PARENT_OVERLAP_TOKENS`
- `CHILD_TARGET_TOKENS`
- `CHILD_MAX_TOKENS`
- `CHILD_OVERLAP_TOKENS`
- `ACTIVE_INDEX_VERSION`
- `MAX_HYDRATED_PARENTS`
- `DOCSTORE_DIR`
- `DocStoreConfig.DB_PATH`

兼容旧代码时：

- `CHUNK_SIZE` 映射到 `CHILD_TARGET_TOKENS`
- `CHUNK_OVERLAP` 映射到 `CHILD_OVERLAP_TOKENS`

## 4. 迁移策略

- 旧 collection 不做在线替换
- 新版本默认写入 `v2` collection 和 `parents_v2.sqlite`
- 文档重建索引时，先删除同 `doc_id` 的 v2 数据，再写入新 parent / child 数据
- 本次未实现在线双库查询和灰度切流

## 5. 测试与验收

已补充并通过的关键测试：

- `ChunkingResult` 返回 parent / child 双层结果
- `list_documents()` 暴露 `parent_count` / `child_count`
- `ParentChildHybridRetriever` 能聚合 child 并回填 parent
- `format_docs_for_context()` 按 parent 文档展示 `page_range`
- `tests/test_retrieval_evals.py` 覆盖 RAGAS 数据转换、dry-run 报告生成和旧自定义指标移除
- 原有 `tests/test_rag_pipeline.py` 全量非慢测试保持通过

当前验收基线：

- `pytest tests/ -q -m 'not slow'`
- `python -m evals.run --dry-run`
- 真实调参对比使用 `python -m evals.run`，并按 `05_rag_ragas_evaluation_design.md` 中的 RAGAS 指标解读结果

## 6. 后续建议

下一阶段优先事项：

1. 为 `section` 切分引入更强的标题层级识别
2. 扩展 `evals/dataset.jsonl` 到 80-120 条，并按概念题、术语题、跨页题、时间题分组看 RAGAS 指标
3. 增加增量重建与内容哈希变更检测
4. 视文档质量决定是否补充 OCR / 表格专项处理
