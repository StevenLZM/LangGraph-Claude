# 01_RAG Elasticsearch 百万级混合检索教学设计

## 1. 文档状态

- 日期：2026-07-26
- 状态：设计已确认，尚未修改业务代码
- 适用模块：`01_RAG`
- 目标读者：希望理解生产级 RAG 检索链路、父子 Chunk、混合检索、排序、权限隔离和引用约束的开发者

本文既是 `01_RAG` 从“Milvus Lite + 进程内 BM25”迁移到 Elasticsearch 的设计规格，也是用于系统设计学习和面试讲解的教学材料。

## 2. 目标

使用“对象存储 + 业务数据库 + Elasticsearch 单检索引擎”作为长期生产架构，其中：

- Elasticsearch 同时承担 Child Chunk 的 BM25 和 Dense kNN 检索。
- BM25 和 Dense 检索使用完全相同的权限及 Metadata Filter。
- 应用层使用 Weighted RRF 融合两个召回列表。
- 保留父子 Chunk：Child 用于召回和排序，Parent 用于最终上下文。
- Cross-Encoder、业务特征、相似去重、文档配额和 MMR 共同控制最终证据质量。
- LLM 只能基于最终证据回答，并必须输出可校验引用。

第一阶段以最小改动迁移检索引擎：

- 继续使用本地文件目录保存 PDF。
- 继续使用 SQLite Parent DocStore。
- 继续使用现有 Embedding、Cross-Encoder、LangChain Chain、Streamlit UI 和评测框架。
- 不自动迁移 Milvus Lite 数据，通过重新摄取文档建立 ES 索引。

对象存储、生产业务数据库和真实登录权限体系属于后续生产化阶段，但本设计提前定义它们与检索层之间的边界。

## 3. 非目标

第一阶段不做以下事项：

- 不安装或启动 Docker 版 Elasticsearch。
- 不修改用户现有 Elasticsearch 全局配置。
- 不引入 Kibana。
- 不使用 Elasticsearch Enterprise 许可下的原生 RRF。
- 不同时保留 Milvus 作为线上检索降级后端。
- 不在没有身份认证来源的情况下宣称已经实现完整 ACL。
- 不凭经验写死 Cross-Encoder 阈值和业务特征权重。
- 不保证引用标识存在就等价于“引用在语义上支持结论”。

## 4. 已确认的本地环境

用户通过以下命令启动现有 Elasticsearch：

```bash
cd ~/Downloads/elasticsearch-8
./bin/elasticsearch
```

2026-07-26 的只读检查结果：

| 项目 | 当前值 | 设计影响 |
|---|---|---|
| Elasticsearch | 8.19.0，tar 安装 | Python 客户端固定在 8.x |
| HTTP 地址 | `http://127.0.0.1:9200` | 本地默认无需 Docker 网络 |
| License | Basic | 使用应用层 RRF |
| Security | 关闭 | 本地无需账号；生产配置仍保留认证能力 |
| HTTP TLS | 关闭 | 本地 `verify_certs=false` |
| 额外插件 | 无 | 第一阶段使用内置 CJK analyzer |
| 节点数 | 1 | 新索引本地副本数设为 0 |
| 集群状态 | yellow | 现有索引副本未分配，主分片正常 |
| 已有索引 | `products_vec`、`products`、空的 `rag-chunks` | 新索引使用独立名称，避免覆盖 |

单节点集群出现 yellow 的原因是现有索引设置了一个副本，而副本不能与主分片分配到同一节点。新建本地 RAG 索引使用一个主分片和零个副本，不要求修改已有索引。

## 5. 当前架构及主要限制

当前 `01_RAG` 的核心链路是：

```text
PDF
  → Parent/Child Chunk
  → Parent 写入 SQLite
  → Child 写入 Milvus Lite

Query
  → Milvus Dense Retrieval
  → 从 Milvus 读取全部 Child，在应用进程构建 BM25
  → LangChain EnsembleRetriever
  → Parent Hydration
  → Cross-Encoder
  → LLM
```

这个实现适合本地演示，但达到百万 Child 规模时存在明显限制：

1. 应用启动或刷新时需要读取大量 Child 构建进程内 BM25。
2. 每个应用实例各自维护 BM25 状态，更新一致性困难。
3. Dense 与 BM25 来自不同执行路径，Filter 语义容易漂移。
4. 进程内 BM25 难以利用 Elasticsearch 的倒排索引、跳表和 WAND 优化。
5. Milvus Lite 更适合单机原型，不是当前目标中的统一生产检索引擎。
6. 父子去重、相似去重、文档配额、MMR 和严格引用还没有形成完整的统一后处理层。

## 6. 关键架构决策

### 6.1 为什么 BM25 和 Dense 都放入 Elasticsearch

统一检索引擎的价值不只是减少一个组件，更重要的是统一以下语义：

- 文档是否 active。
- 用户是否有权限访问。
- 文档版本和有效期是否合法。
- 时间、类型、语言和租户过滤是否一致。
- 删除和更新是否同时影响两个召回通道。

BM25 和 Dense 仍然是两个独立排名列表，但它们读取相同 Child 文档、相同 Metadata，并使用同一个 Filter Builder。

### 6.2 为什么保留应用层 RRF

本地 Elasticsearch 使用 Basic License，而当前方案需要可长期免费运行。应用层 Weighted RRF：

- 不依赖 ES Enterprise RRF 许可。
- 保留当前 LangChain `EnsembleRetriever` 的迁移路径。
- 可以配置 BM25 和 Dense 权重。
- 可以显式使用 `child_id` 作为融合身份。
- 便于记录每个召回通道的排名和调试信息。

RRF 公式为：

```text
rrf_score(d) = Σ weight_i / (rrf_k + rank_i(d))
```

其中 `rank_i` 从 1 开始，默认 `rrf_k=60`。RRF 使用排名而不是原始 `_score`，避免 BM25 分数和向量相似度不在同一量纲的问题。

### 6.3 为什么 Child 检索、Parent 生成

小 Child 更容易精准匹配用户问题，大 Parent 更适合给 LLM 提供完整上下文。

| 对象 | 主要职责 | 是否写入 ES | 是否进入最终 Prompt |
|---|---|---:|---:|
| Child Chunk | BM25、Dense、RRF、Cross-Encoder、引用锚点 | 是 | 作为命中证据标记 |
| Parent Chunk | 完整语义上下文、Token Budget 组装 | 第一阶段否 | 是 |

最终的“5～8 个 Chunk”指去重后的 Parent Chunk；每个 Parent 携带命中的 Child 作为引用证据。

## 7. 目标查询架构

```text
用户 Query
  ↓
Query 标准化、实体识别、意图识别
  ↓
可信身份上下文 + 权限与 Metadata Filter
  ↓
┌────────────────────────┬────────────────────────┐
│ Elasticsearch BM25     │ Elasticsearch Dense kNN│
│ Top 50                 │ Top 50 / HNSW           │
└────────────────────────┴────────────────────────┘
  ↓
各通道内部按 child_id 精确去重
每个 parent 暂时最多保留 2～3 个 Child
  ↓
应用层 Weighted RRF，以 child_id 为融合主键
保留约 60～80 个 Child
  ↓
Cross-Encoder：80 → 15
  ↓
融合时间、权威性、版本等业务特征
  ↓
按 parent_id 聚合
  ↓
完全重复与相似 Parent 去重
doc_id 配额 + MMR 多样性
  ↓
Token Budget 选择 5～8 个 Parent
  ↓
LLM 回答并引用具体 Child 证据
  ↓
引用合法性校验
```

## 8. Query 分析与 RetrievalContext

### 8.1 Query 标准化

标准化只做不改变业务语义的确定性操作：

- Unicode 规范化。
- 合并多余空白。
- 统一明显的全角/半角形式。
- 保留版本号、产品型号、日期和专有名词。
- 记录原始 Query，不覆盖审计信息。

不应盲目删除停用词或数字，因为“不得”“不是”“2024 版”等信息可能决定答案。

### 8.2 实体和意图识别

建议输出结构化结果：

```python
QueryAnalysisResult(
    original_query=...,
    normalized_query=...,
    entities={
        "product": [...],
        "version": [...],
        "date": [...],
        "document_type": [...],
    },
    intent="current_policy | historical | troubleshooting | factual | comparison",
    time_range=...,
)
```

实体识别结果只能用于构造业务查询条件，不能生成或提升用户权限。高风险字段必须使用白名单映射，不能把模型生成的任意字段名直接拼入 ES DSL。

### 8.3 可信 RetrievalContext

```python
RetrievalContext(
    query_analysis=...,
    tenant_id=...,
    allowed_principals=...,
    visibility=...,
    metadata_filters=...,
    trace_id=...,
)
```

`tenant_id`、`allowed_principals` 和用户角色必须来自可信认证系统，而不是 Query 或 LLM。

## 9. 权限与 Metadata Filter

Filter 必须在召回前下推到 ES，不能先召回越权文档再在应用层删除。

典型逻辑：

```text
status = active
AND tenant_id = 当前租户
AND (
    visibility = public
    OR acl_principals 与当前用户主体有交集
)
AND 文档类型、标签、时间、版本等业务 Filter
```

相同 Filter 对象同时传给 BM25 和 kNN 查询。缺少可信身份时采用 fail-closed：

- 只允许查询明确标记为 public 的内容。
- 不把空权限数组解释为“允许全部”。
- Filter 构造失败时终止检索，不退化为无 Filter 查询。

第一阶段尚无真实登录体系时，只实现 Filter 数据结构、公开文档默认值和离线测试；不能把占位字段描述成完整权限能力。

## 10. Elasticsearch Child 索引

### 10.1 索引和别名

```text
物理索引：rag-child-chunks-v1
读别名：  rag-child-chunks-read
写别名：  rag-child-chunks-write
```

版本化物理索引支持未来修改 analyzer、向量维度或 Mapping 后重建索引，再原子切换读别名。

### 10.2 字段设计

| 类别 | 字段 | ES 类型及用途 |
|---|---|---|
| 标识 | `storage_id` | ES `_id`，格式为 `doc_id:doc_version:child_id` |
| 标识 | `child_id` | `keyword`，作为 RRF 的逻辑融合身份 |
| 标识 | `parent_id`、`doc_id` | `keyword`，聚合、删除和 Parent Hydration |
| 内容 | `content` | `text`，BM25 |
| 向量 | `embedding` | `dense_vector`，cosine + HNSW |
| 定位 | `source`、`file_name` | `keyword` |
| 定位 | `page_number`、`chunk_index` | `integer` |
| 权限 | `tenant_id`、`acl_principals`、`visibility` | `keyword` |
| 状态 | `status`、`ingest_run_id` | `keyword` |
| 分类 | `document_type`、`language`、`tags` | `keyword` |
| 时间 | `created_at`、`updated_at`、`valid_from`、`valid_to` | `date` |
| 现有时间语义 | `upload_date`、`doc_date_min`、`doc_date_max`、`has_doc_date` | 与现有逻辑兼容 |
| 业务特征 | `version` | `keyword` |
| 业务特征 | `version_rank` | `integer` |
| 业务特征 | `authority_score` | `float`，约束在 `[0,1]` |
| 幂等和追踪 | `content_hash`、`embedding_model`、`embedding_version` | `keyword` |

Mapping 应明确声明向量维度，并在应用启动或索引初始化时校验：

```text
实际 Query Embedding 维度 == 索引 Mapping dims
```

不允许维度错误到第一次用户查询时才暴露。

### 10.3 为什么 ES `_id` 不能直接等于逻辑 `child_id`

当前 `child_id` 基于 Child 内容生成。同一段未变化的内容可能同时出现在文档的新旧版本中。如果直接使用 `child_id` 作为 ES `_id`：

1. 写入 staging 新版本时会覆盖旧版本中仍 active 的同一 Child。
2. 新版本写入失败后清理 staging 数据，可能把旧版本的有效 Child 一并删除。
3. “新版本先校验、再激活、最后删除旧版本”的无损切换无法成立。

因此需要分离两个身份：

```text
逻辑检索身份 child_id
  → 用于 BM25/Dense 结果匹配、RRF 和引用

物理存储身份 storage_id = doc_id:doc_version:child_id
  → 用作 ES _id，允许新旧版本在切换期间短暂共存
```

同一文档版本重试时 `storage_id` 不变，仍然保持幂等；不同版本不会互相覆盖。

### 10.4 中文 analyzer

本地 ES 没有安装额外插件。第一阶段使用 ES 内置 `cjk` analyzer，保证无需改变用户安装。

生产选择 analyzer 时应使用中文检索评测集比较：

- 内置 CJK analyzer。
- IK 等第三方分词器。
- 业务词典和同义词方案。

更复杂的分词器不天然代表效果更好；插件版本兼容、升级和词典发布也会增加运维成本。

### 10.5 本地与生产索引参数

本地默认：

```text
number_of_shards=1
number_of_replicas=0
```

生产分片数不能照搬固定值，应根据 Child 数量、向量维度、索引体积、节点内存、查询并发和恢复时间压测后决定。生产至少需要副本来提供节点故障恢复能力。

## 11. 检索参数

初始生产目标值全部通过环境变量配置：

```text
BM25_TOP_K=50
DENSE_TOP_K=50
DENSE_NUM_CANDIDATES=300
MAX_CHILDREN_PER_PARENT_PRE_RRF=3
RRF_TOP_K=80
RRF_K=60
RERANK_TOP_K=15
FINAL_PARENT_TOP_K=6
MAX_PARENTS_PER_DOCUMENT=2
```

这些是起始值而不是普适最优值：

- 增大 `DENSE_NUM_CANDIDATES` 通常提高 ANN Recall，但增加延迟和 CPU。
- 增大初始 Top K 提高召回机会，但增加 Rerank 成本。
- `RRF_K` 控制头部排名差异的敏感度。
- 最终数量必须同时受 Token Budget 约束。

本地机器可以通过 `.env` 调低候选数，但不能在代码里建立另一套流程。

## 12. 去重与 RRF 的正确顺序

不能在 RRF 前按 `parent_id` 完全去重。

原因：

1. 同一个 Parent 可能有多个 Child 分别命中关键词和语义。
2. 如果先只保留一个 Child，会丢失另一个通道的排名贡献。
3. RRF 需要看到两个有序列表中相同 `child_id` 的位置，才能累加证据。

推荐顺序：

1. 每个召回通道内部按 `child_id` 删除精确重复项。
2. 每个通道对单个 Parent 暂时限额 2～3 个 Child，防止一个长文档淹没候选。
3. RRF 以 `child_id` 为身份融合两个排名列表。
4. Cross-Encoder 之后才进行完整 Parent 聚合和最终文档多样性控制。

LangChain `EnsembleRetriever` 必须显式使用 `child_id`，或由适配层保证身份键是 `child_id`。不能默认用全文内容作为生产身份键。

## 13. Cross-Encoder 与业务特征

### 13.1 Cross-Encoder

RRF Top 80 进入 Cross-Encoder，保留 Top 15。Cross-Encoder 同时读取 Query 和 Child 文本，负责比双塔向量模型更精细地判断相关性。

Cross-Encoder 原始输出可能是 logit，也可能是特定模型定义的分数。应用必须先明确模型输出语义，再进行归一化。

### 13.2 阈值

`rerank_score` 阈值不能凭经验硬编码：

1. 建立包含相关和不相关候选的标注集。
2. 收集分数分布。
3. 绘制 Precision/Recall 曲线。
4. 根据业务对错误回答和拒答的成本选择阈值。
5. 将阈值写入配置，并记录模型版本。

低于阈值的候选不进入 LLM 上下文。如果全部候选均低于阈值，系统返回“当前知识库证据不足”，而不是自动放宽阈值。

### 13.3 硬规则和软特征

以下条件应当作为硬 Filter：

- 权限。
- `status=active`。
- 已失效且不允许历史查询的版本。
- 明确的有效期。
- 租户隔离。

以下信息适合作为软特征：

- Cross-Encoder 相关性。
- RRF 排名。
- 来源权威性。
- 根据意图启用的新鲜度。
- 版本优先级。

初始教学公式：

```text
final_score =
    0.70 × normalized_rerank_score
  + 0.15 × normalized_rrf_score
  + 0.07 × authority_score
  + 0.05 × freshness_score
  + 0.03 × version_score
```

该公式只用于给出可解释的起点，所有权重必须配置化并通过评测调优。

新鲜度不是全局奖励。历史查询应关闭或反转新鲜度偏好，否则系统会错误地用新文档回答旧版本问题。

## 14. Parent 聚合、相似去重与 MMR

Cross-Encoder Top 15 Child 按 `parent_id` 聚合：

```text
parent_score =
    最佳 Child final_score
  + capped_multi_evidence_bonus
```

额外 Child 只能产生封顶的小幅证据加成，避免长 Parent 因 Child 数量多而占优势。

最终选择步骤：

1. 按 `content_hash` 删除完全相同 Parent。
2. 使用代表 Child 或 Parent 向量删除近似重复内容。
3. 每个 `doc_id` 最多保留默认两个 Parent。
4. 使用 MMR 平衡相关性和候选间差异。

MMR 概念公式：

```text
MMR(candidate) =
    λ × relevance(candidate)
  - (1 - λ) × max_similarity(candidate, selected)
```

初始配置：

```text
SIMILARITY_DEDUP_THRESHOLD=0.92
MMR_LAMBDA=0.70
MAX_PARENTS_PER_DOCUMENT=2
```

阈值和 `λ` 同样需要评测。技术文档中重复的页眉、模板和版本说明可能产生很高相似度，测试集应覆盖这类情况。

## 15. Token Budget 与上下文组装

最终 5～8 个 Parent 是数量目标，Token Budget 才是硬约束：

```text
模型上下文窗口
  - System Prompt
  - 用户问题
  - 需要保留的对话历史
  - 回答输出预留
  - 安全余量
  = 证据上下文预算
```

ParentContextAssembler 按最终分数依次装入 Parent：

- 同一 `parent_id` 只加载一次。
- 保存标题、来源、页码、版本和命中 Child 信息。
- 超出预算时停止添加低排名 Parent。
- 必须裁剪时在句子或段落边界处理。
- 不允许裁掉引用标识与来源 Metadata。
- 不为了凑够五个 Parent 而突破预算。

如果一个 Parent 本身过长，长期方案应调整 Parent Chunk 设计，而不是依赖任意字符截断。

## 16. 引用约束

最终证据分配稳定的 Prompt 内标识：

```text
[S1] → doc_id / parent_id / child_id / source / page / version
[S2] → ...
```

Prompt 规则：

- 所有来自知识库的事实性结论必须携带一个或多个 `[Sx]`。
- 不得引用未提供的来源。
- 证据不足时明确说明，不使用常识补齐业务事实。
- 答案末尾输出来源列表。

生成后校验：

1. 解析回答中的引用。
2. 验证每个引用都在最终证据集合中。
3. 检查需要引用的回答是否至少包含一个合法引用。
4. 无引用或存在非法引用时，执行一次受约束重新生成。
5. 第二次仍失败时返回安全答案和可用证据列表。

必须区分三个概念：

| 指标 | 问题 |
|---|---|
| 引用合法性 | `[S1]` 是否真的存在于证据集合 |
| 引用完整性 | 应引用的事实是否都带引用 |
| 语义忠实度 | 引用内容是否真正支持对应结论 |

正则或 ID 校验只能解决第一项。语义忠实度需要人工标注、LLM-as-judge 或 NLI/事实一致性评测，并应抽样人工复核。

## 17. 写入一致性

SQLite 和 Elasticsearch 之间不存在分布式事务。设计目标是让不一致状态“不可检索且可修复”。

推荐写入流程：

```text
解析文档
  ↓
生成稳定 doc_id、parent_id、child_id
  ↓
写入 Parent SQLite
  ↓
批量生成 Embedding
  ↓
ES Bulk 写入 status=staging 的 Child
  ↓
校验预期数量和每个 Bulk Item
  ↓
将本次 ingest_run_id 切换为 active
  ↓
清理旧版本
```

关键规则：

- `storage_id=doc_id:doc_version:child_id` 作为 ES `_id`；同一版本重试时覆盖而不是重复创建，不同版本可以在切换期间共存。
- 查询只读取 `status=active`。
- Bulk API 的 HTTP 成功不代表每个 Item 都成功，必须逐项检查错误。
- 部分失败时删除本次 `ingest_run_id` 的 staging 数据。
- 激活新版本后再清理旧版本，减少更新期间的不可用窗口。
- Parent 写入失败时不写入可检索 Child。
- Child 激活失败时 Parent 可以暂时成为不可检索孤儿，但不能出现可检索 Child 指向不存在 Parent。

删除顺序与写入相反：

1. 先将 ES Child 设为不可检索或删除。
2. 再删除 Parent。
3. 生产环境优先软删除并异步物理清理。

应提供周期性一致性检查：

- active Child 找不到 Parent。
- Parent 长期没有 active Child。
- 同一个 `doc_id` 存在多个错误激活版本。
- staging 数据超过允许时长。

## 18. 查询降级与失败处理

| 故障 | 行为 | 原因 |
|---|---|---|
| 权限上下文缺失 | 只查公开内容或拒绝请求 | fail-closed |
| Query 分析失败 | 使用原始 Query 和最小安全 Filter | 标准化不应成为单点故障 |
| Embedding 失败 | 降级为 BM25 | 关键词检索仍可提供证据 |
| Dense 超时 | 使用 BM25 列表 | 单路失败不必终止整个查询 |
| BM25 超时 | 使用 Dense 列表 | 同上 |
| Cross-Encoder 失败 | 使用 RRF + 业务特征 | 保留基础相关性 |
| Parent 缺失 | 丢弃候选并记录一致性告警 | 不能向 LLM 传空上下文 |
| ES 整体不可用 | 返回检索服务不可用 | 不静默读取旧 Milvus 数据 |
| 全部证据低于阈值 | 返回证据不足 | 不用低质量证据强行回答 |
| 引用校验失败 | 一次重试，仍失败则安全返回 | 防止伪造引用 |

降级结果必须带内部 Trace 标记，不能让运维和评测把降级答案误认为正常链路。

## 19. 组件边界与最小改动

### 19.1 目标组件

```text
QueryAnalyzer
  → QueryAnalysisResult

RetrievalFilterBuilder
  → 统一 ES bool/filter DSL

ElasticsearchBM25Retriever
  → Top 50 Child

ElasticsearchDenseRetriever
  → Query Embedding + kNN Top 50 Child

HybridFusionRetriever
  → 并行召回 + Weighted RRF

BusinessReranker
  → Cross-Encoder + 业务特征

DiversityFilter
  → 精确去重 + 近似去重 + 文档配额 + MMR

ParentContextAssembler
  → Parent Hydration + Token Budget

CitationValidator
  → 引用合法性检查和受约束重试
```

### 19.2 第一阶段预计修改

- `01_RAG/config.py`
- `01_RAG/rag/vectorstore.py`
- `01_RAG/rag/retriever.py`
- `01_RAG/requirements.txt`
- `01_RAG/.env.example`
- `01_RAG/README.md`
- `01_RAG/tests/` 下相关离线测试

预计新增：

- Elasticsearch Child Store 或 Retriever 适配模块。
- 后处理、多样性或引用校验模块。
- ES Mapping/索引初始化脚本。

尽量保持以下边界不变：

- `rag/chain.py` 的主调用方式。
- Streamlit 页面。
- Parent SQLite DocStore。
- PDF Loader 和现有父子切分逻辑。
- Embedding Factory 和 Cross-Encoder Factory。

不新增 Docker Compose。

## 20. 配置设计

本地基础配置：

```env
ES_URL=http://127.0.0.1:9200
ES_INDEX_READ_ALIAS=rag-child-chunks-read
ES_INDEX_WRITE_ALIAS=rag-child-chunks-write
ES_NUMBER_OF_SHARDS=1
ES_NUMBER_OF_REPLICAS=0
ES_VERIFY_CERTS=false
```

生产配置还应支持：

```text
ES_USERNAME / ES_PASSWORD
或 ES_API_KEY
ES_CA_CERTS
ES_REQUEST_TIMEOUT
ES_MAX_RETRIES
ES_BULK_CHUNK_SIZE
ES_EMBEDDING_DIMS
```

本地关闭安全并不意味着代码可以假设生产也关闭安全。认证、TLS 和证书验证必须通过配置启用，生产默认不得关闭证书验证。

Python 依赖固定为：

```text
elasticsearch>=8.19,<9
```

项目按照仓库规范使用 Python 3.11+。不能因为当前系统存在 Python 3.9 就降低项目运行基线。

## 21. 观测性

每次查询至少记录：

- `trace_id`。
- Query 分析耗时。
- ES BM25、Dense 各自耗时和命中数。
- Dense `num_candidates` 和 Top K。
- RRF 前后的候选数。
- Cross-Encoder 耗时、分数分布和阈值过滤数量。
- Parent 聚合、相似去重、文档配额和 MMR 各删除多少候选。
- 最终 Parent 数量和 Token 数。
- 是否发生降级及原因。
- 最终证据 ID，不记录不必要的敏感正文。

建议关注的生产指标：

- 各阶段 P50/P95/P99 延迟。
- ES 超时和错误率。
- Embedding 与 Reranker 错误率。
- Parent 缺失率。
- 无证据回答率。
- 降级率。
- 索引吞吐量和 Bulk 失败率。

## 22. 测试和评测

### 22.1 单元测试

- Query 标准化不破坏版本号和否定词。
- 权限 Filter 缺失时 fail-closed。
- BM25 和 Dense 使用同一个 Filter。
- Weighted RRF 使用 `child_id`，公式和排序可重复。
- RRF 前不会按 `parent_id` 完全去重。
- Cross-Encoder 阈值过滤。
- Parent 分数组合和证据加成封顶。
- 完全重复及相似内容去重。
- 每文档 Parent 配额。
- MMR 相关性/多样性选择。
- Token Budget 不溢出。
- 引用合法、非法、缺失和重试路径。

### 22.2 Elasticsearch 集成测试

- 创建版本化测试索引和别名。
- BM25 可以召回关键词样本。
- kNN 可以召回语义样本。
- 权限、时间和版本 Filter 对两路查询一致。
- Bulk 部分失败能够被识别和补偿。
- staging 文档不能被查询。
- 删除 `doc_id` 后不再召回对应 Child。

集成测试必须使用独立测试索引前缀，不能操作用户已有的 `products_vec`、`products` 或 `rag-chunks`。

### 22.3 离线检索指标

- BM25 Recall@50。
- Dense Recall@50。
- Fusion Recall@80。
- Cross-Encoder NDCG@15。
- MRR。
- 最终 Parent Recall。
- 不同文档类型、语言、时间意图和版本问题的分桶指标。

### 22.4 答案和引用指标

- 引用合法率。
- 引用覆盖率。
- 引用语义准确率。
- Faithfulness / Groundedness。
- 无证据时的正确拒答率。
- 权限泄漏数量，验收要求必须为零。

### 22.5 性能和容量

- 单文档和批量索引吞吐。
- 百万 Child 下的索引体积。
- BM25、kNN、RRF、Rerank 和端到端 P95。
- 不同 `num_candidates` 对 Recall/Latency 的曲线。
- 并发查询下的堆内存、CPU 和 GC。

所有效果指标都应与当前“Milvus Dense + 进程内 BM25”建立基线比较，避免只验证功能可运行，却忽略召回质量下降。

## 23. 分阶段落地

### 阶段 A：ES 检索等价迁移

- 建立 Child 索引、Mapping 和别名。
- ES 完成 BM25 和 Dense kNN。
- 保持 Parent SQLite。
- 应用层 Weighted RRF。
- 保留现有时间 Filter、Cross-Encoder 和 Chain 接口。
- 重建文档索引。

目标：去掉 Milvus Lite 和进程内全量 BM25，不改变最终产品交互。

### 阶段 B：生产级后处理

- QueryAnalysisResult 和统一 RetrievalContext。
- Cross-Encoder 阈值校准。
- 业务特征融合。
- Parent 聚合、相似去重、每文档配额和 MMR。
- Token Budget 和引用校验。
- 完整观测指标及离线评测。

目标：从“混合召回可用”升级到“最终证据可控”。

### 阶段 C：生产存储与权限

- PDF/原文件迁移到对象存储。
- Parent 正文和业务状态迁移到业务数据库或适合的大对象存储方案。
- 接入真实认证和 ACL。
- 多节点 ES、TLS、认证、备份、容量规划和告警。
- 在线版本发布和一致性修复任务。

目标：支持多租户、百万级数据和生产运维。

## 24. 典型错误及教学要点

### 错误 1：把 BM25 分数和余弦相似度直接相加

两种分数的量纲和分布不同。RRF 使用排名融合，避免未经校准的原始分数相加。

### 错误 2：RRF 前按 Parent 完全去重

会丢失不同 Child 和不同召回通道的证据贡献。先以 Child 融合，再聚合 Parent。

### 错误 3：权限在召回后过滤

越权文档已经进入候选、日志或缓存，存在泄漏风险。权限必须下推到每个召回通道。

### 错误 4：只配置 Top K，不配置 `num_candidates`

ANN 的候选探索宽度影响 Recall。只关注最终 K 会忽略 HNSW Recall/Latency 的核心调节项。

### 错误 5：Cross-Encoder Top 15 直接全部交给 LLM

候选之间可能高度重复或来自同一文档。还需要 Parent 聚合、近似去重、配额、MMR 和 Token Budget。

### 错误 6：Prompt 写了“必须引用”就认为不会伪造

Prompt 是软约束。必须在生成后校验引用 ID，并通过评测验证语义忠实度。

### 错误 7：单节点 yellow 就认为数据不可用

需要区分未分配副本和未分配主分片。本地单节点可将新索引副本设为零；生产不能照搬。

### 错误 8：为了高可用静默切回旧 Milvus

旧索引可能已经不再同步，静默降级会返回过期或越权结果。双检索引擎降级需要额外的一致性协议，本阶段明确不做。

## 25. 验收标准

设计落地后应满足：

1. BM25 和 Dense 都从 Elasticsearch 8.19 查询。
2. 应用运行不再需要 Milvus Lite，也不加载全部 Child 构建 BM25。
3. 两个召回通道使用同一个权限及 Metadata Filter。
4. RRF 以 `child_id` 为融合键，参数可配置。
5. 检索 Child、生成 Parent、引用 Child 的职责可追踪。
6. Cross-Encoder 阈值、业务权重、去重阈值和 MMR 参数可配置。
7. 每个文档的最终 Parent 数量受配额控制。
8. 最终上下文不超过 Token Budget。
9. 非法引用不能直接返回用户。
10. 权限测试中的越权召回为零。
11. 单路召回或 Reranker 故障时有明确降级记录。
12. ES 整体不可用时不静默切回旧 Milvus。
13. 离线效果不低于迁移前基线，性能指标达到经评审的目标。

## 26. 参考资料

- [Elasticsearch dense vector](https://www.elastic.co/docs/reference/elasticsearch/mapping-reference/dense-vector/)
- [Elasticsearch retrievers overview](https://www.elastic.co/docs/solutions/search/retrievers-overview)
- [Elasticsearch RRF retriever](https://www.elastic.co/docs/reference/elasticsearch/rest-apis/retrievers/rrf-retriever)
- [Elasticsearch CJK bigram token filter](https://www.elastic.co/docs/reference/text-analysis/analysis-cjk-bigram-tokenfilter)
- [Elasticsearch Python client](https://www.elastic.co/docs/reference/elasticsearch/clients/python)
- [Elastic subscriptions](https://www.elastic.co/subscriptions)
