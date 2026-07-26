# 人工评测标注流程

生产评测只接受真实语料和双人复核的 qrels，不允许由脚本、LLM 或演示数据自动补齐标签。

1. 从真实业务场景选择一个 Query。
2. 在当前索引语料中定位 Source、Page 和 Section。
3. 复制一段短小、原子化且能独立支撑答案的 `evidence_text`。
4. 按 `0..3` 标注相关性：`0` 不相关、`1` 弱相关、`2` 可支撑、`3` 核心证据。
5. 由第二位标注者复核 Reference、证据锚点和 Grade。
6. 按业务实体或文档组将通过复核的 Case 放入且仅放入 Train、Dev、Test 之一。
7. 重新计算 `manifest.json` 的 Case 数量和文件 SHA-256。

`answer` Case 必须至少有一个 Grade 不低于 2 的 qrel。`abstain` Case 可以没有 qrel。`deny` Case 可记录受限证据 qrel，用于检测权限泄漏，但不计入普通 Recall/NDCG。

旧版单文件评测集的五条内容只属于待复核种子，保存在 `legacy_seed_cases.jsonl`；未经上述流程不得进入 Dev 或 Test。
