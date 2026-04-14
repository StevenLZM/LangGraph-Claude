"""
generate_sample_pdfs.py
生成两份用于 RAG 演示的示例 PDF 文档：
  1. AI Agent 技术白皮书（中文，10页）
  2. LangGraph 开发手册（中文，8页）
"""
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, HRFlowable
)
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import os, platform

# ── 字体注册（跨平台中文支持） ────────────────────────────────────
def register_chinese_font():
    system = platform.system()
    font_candidates = {
        "Darwin": [
            "/System/Library/Fonts/STHeiti Light.ttc",
            "/System/Library/Fonts/PingFang.ttc",
            "/Library/Fonts/Arial Unicode MS.ttf",
            "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        ],
        "Linux": [
            "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        ],
        "Windows": [
            "C:/Windows/Fonts/simhei.ttf",
            "C:/Windows/Fonts/simsun.ttc",
        ],
    }
    for path in font_candidates.get(system, []):
        if os.path.exists(path):
            try:
                pdfmetrics.registerFont(TTFont("ChineseFont", path))
                return "ChineseFont"
            except Exception:
                continue
    # 兜底：使用 Helvetica（不支持中文，但不崩溃）
    return "Helvetica"

FONT_NAME = register_chinese_font()


def make_styles():
    """创建文档样式"""
    base = getSampleStyleSheet()
    styles = {
        "title": ParagraphStyle(
            "DocTitle", fontName=FONT_NAME, fontSize=22,
            textColor=colors.HexColor("#1a1a2e"), spaceAfter=12,
            alignment=TA_CENTER, leading=28,
        ),
        "subtitle": ParagraphStyle(
            "Subtitle", fontName=FONT_NAME, fontSize=13,
            textColor=colors.HexColor("#4a4a8a"), spaceAfter=8,
            alignment=TA_CENTER, leading=18,
        ),
        "h1": ParagraphStyle(
            "H1", fontName=FONT_NAME, fontSize=16,
            textColor=colors.HexColor("#1a1a2e"), spaceBefore=16,
            spaceAfter=8, leading=22,
        ),
        "h2": ParagraphStyle(
            "H2", fontName=FONT_NAME, fontSize=13,
            textColor=colors.HexColor("#2d2d6b"), spaceBefore=12,
            spaceAfter=6, leading=18,
        ),
        "body": ParagraphStyle(
            "Body", fontName=FONT_NAME, fontSize=10,
            textColor=colors.HexColor("#333333"), spaceAfter=6,
            leading=16, alignment=TA_JUSTIFY,
        ),
        "bullet": ParagraphStyle(
            "Bullet", fontName=FONT_NAME, fontSize=10,
            textColor=colors.HexColor("#333333"), spaceAfter=4,
            leading=15, leftIndent=20,
        ),
        "caption": ParagraphStyle(
            "Caption", fontName=FONT_NAME, fontSize=9,
            textColor=colors.HexColor("#888888"), spaceAfter=4,
            alignment=TA_CENTER, leading=13,
        ),
        "code": ParagraphStyle(
            "Code", fontName="Courier", fontSize=9,
            textColor=colors.HexColor("#1a1a2e"),
            backColor=colors.HexColor("#f5f5f5"),
            spaceAfter=6, leading=13, leftIndent=10, rightIndent=10,
        ),
    }
    return styles


# ════════════════════════════════════════════════════════════════
# 文档 1：AI Agent 技术白皮书
# ════════════════════════════════════════════════════════════════
def build_ai_agent_whitepaper(output_path: str):
    doc = SimpleDocTemplate(
        output_path, pagesize=A4,
        leftMargin=2.5*cm, rightMargin=2.5*cm,
        topMargin=2.5*cm, bottomMargin=2.5*cm,
    )
    s = make_styles()
    story = []

    # ── 封面 ──────────────────────────────────────────────────────
    story += [
        Spacer(1, 3*cm),
        Paragraph("AI Agent 技术白皮书", s["title"]),
        Paragraph("企业智能化转型的核心技术框架", s["subtitle"]),
        Spacer(1, 0.5*cm),
        HRFlowable(width="80%", thickness=2, color=colors.HexColor("#6C63FF"), spaceAfter=20),
        Paragraph("版本 2.0 · 2026年4月", s["caption"]),
        Paragraph("LangGraph-Claude 技术团队 编著", s["caption"]),
        PageBreak(),
    ]

    # ── 第1章 AI Agent 概述 ───────────────────────────────────────
    story += [
        Paragraph("第一章  AI Agent 概述", s["h1"]),
        HRFlowable(width="100%", thickness=1, color=colors.HexColor("#e0e0e0"), spaceAfter=8),

        Paragraph("1.1 什么是 AI Agent", s["h2"]),
        Paragraph(
            "AI Agent（人工智能代理）是一种能够感知环境、制定决策并采取行动以实现特定目标的智能系统。"
            "与传统的问答系统不同，AI Agent 具备自主推理、工具调用和多步骤规划能力。"
            "它可以将复杂任务分解为子任务，循环执行直至达成目标。", s["body"]
        ),
        Paragraph(
            "AI Agent 的三大核心能力：", s["body"]
        ),
        Paragraph("• 感知（Perception）：通过多种输入源获取环境信息，包括文本、图像、结构化数据等。", s["bullet"]),
        Paragraph("• 推理（Reasoning）：基于大语言模型进行逻辑推理，分解问题，制定行动计划。", s["bullet"]),
        Paragraph("• 行动（Action）：调用外部工具（搜索、计算、API）执行具体操作并获取反馈。", s["bullet"]),
        Spacer(1, 0.3*cm),

        Paragraph("1.2 Agent 与普通 LLM 应用的区别", s["h2"]),
        Paragraph(
            "普通 LLM 应用采用单次问答模式（输入→输出），而 Agent 采用循环推理模式（思考→行动→观察→再思考）。"
            "这种循环使得 Agent 能够处理需要多步骤推理的复杂任务，如：调研报告生成、代码调试、"
            "数据分析等。Agent 的行动空间由其配置的工具集决定，可以无限扩展。", s["body"]
        ),
        Spacer(1, 0.3*cm),

        Paragraph("1.3 Agent 的主要架构类型", s["h2"]),
        Paragraph(
            "目前业界主流的 Agent 架构类型分为以下四种：", s["body"]
        ),
        Paragraph("• ReAct（Reasoning + Acting）：交替进行推理和行动，是最经典的单 Agent 模式，适合工具调用场景。", s["bullet"]),
        Paragraph("• Plan-and-Execute：先制定完整计划，再逐步执行，适合长任务规划场景。", s["bullet"]),
        Paragraph("• Multi-Agent：多个专业化 Agent 协作，通过消息传递和状态共享完成复杂任务。", s["bullet"]),
        Paragraph("• Human-in-the-Loop：在关键决策节点引入人工审核，适合高风险业务场景。", s["bullet"]),
        PageBreak(),
    ]

    # ── 第2章 RAG 技术详解 ────────────────────────────────────────
    story += [
        Paragraph("第二章  RAG 技术详解", s["h1"]),
        HRFlowable(width="100%", thickness=1, color=colors.HexColor("#e0e0e0"), spaceAfter=8),

        Paragraph("2.1 RAG 的定义与原理", s["h2"]),
        Paragraph(
            "RAG（Retrieval-Augmented Generation，检索增强生成）是一种将信息检索与文本生成相结合的技术框架。"
            "其核心思想是：在 LLM 生成回答之前，先从外部知识库中检索相关文档，将检索结果作为上下文注入 Prompt，"
            "从而使 LLM 能够基于最新、准确的私有知识回答问题，有效解决 LLM 的幻觉问题和知识截止问题。", s["body"]
        ),

        Paragraph("2.2 RAG 完整流程", s["h2"]),
        Paragraph(
            "RAG 系统分为两个主要阶段：索引阶段（Indexing）和检索生成阶段（Retrieval & Generation）。", s["body"]
        ),
        Paragraph("索引阶段步骤：", s["body"]),
        Paragraph("① 文档加载（Document Loading）：支持 PDF、Word、网页、数据库等多种数据源。", s["bullet"]),
        Paragraph("② 文本分块（Text Splitting）：将长文档切割为合适大小的文本块（Chunk），通常 200-1000 字符。", s["bullet"]),
        Paragraph("③ 向量化（Embedding）：使用 Embedding 模型将文本块转化为高维向量表示。", s["bullet"]),
        Paragraph("④ 存储（Storage）：将向量及元数据存入向量数据库（如 Chroma、Pinecone、Milvus）。", s["bullet"]),
        Spacer(1, 0.2*cm),
        Paragraph("检索生成阶段步骤：", s["body"]),
        Paragraph("① 问题向量化：将用户问题转化为向量。", s["bullet"]),
        Paragraph("② 相似度检索：在向量库中找出最相关的 Top-K 文档块。", s["bullet"]),
        Paragraph("③ 上下文构建：将检索结果拼接为 Context，注入 Prompt。", s["bullet"]),
        Paragraph("④ LLM 生成：基于 Context 生成准确答案。", s["bullet"]),
        Spacer(1, 0.3*cm),

        Paragraph("2.3 混合检索策略", s["h2"]),
        Paragraph(
            "单纯的语义检索（Dense Retrieval）在某些场景表现不佳，例如精确的关键词匹配、"
            "专业术语查询等。生产环境推荐采用混合检索策略：", s["body"]
        ),
        Paragraph(
            "语义检索（Dense）+ 关键词检索（BM25 Sparse）结合，通过 RRF（Reciprocal Rank Fusion）"
            "算法融合两种检索结果。语义检索捕捉语义相关性，BM25 保证关键词精确匹配，两者互补，"
            "显著提升召回率和准确率。根据实验数据，混合检索相比单一语义检索，准确率平均提升 15-20%。", s["body"]
        ),

        Paragraph("2.4 高级 RAG 技术", s["h2"]),
        Paragraph("当前业界常用的高级 RAG 优化技术包括：", s["body"]),
        Paragraph("• Query Rewriting（问题改写）：多轮对话中，将含代词的问题改写为独立完整问题，提升检索准确性。", s["bullet"]),
        Paragraph("• HyDE（假设文档嵌入）：让 LLM 生成一个假设答案，用该答案的向量检索，而非用问题向量检索。", s["bullet"]),
        Paragraph("• Context Compression（上下文压缩）：从检索到的文档块中只提取与问题直接相关的句子，减少噪声。", s["bullet"]),
        Paragraph("• Reranker（重排序）：用交叉编码器对召回结果重新打分排序，精度优于双塔模型。", s["bullet"]),
        Paragraph("• Self-RAG：让模型自主决定何时检索、检索什么、如何使用检索结果。", s["bullet"]),
        PageBreak(),
    ]

    # ── 第3章 技术选型 ────────────────────────────────────────────
    story += [
        Paragraph("第三章  技术选型与最佳实践", s["h1"]),
        HRFlowable(width="100%", thickness=1, color=colors.HexColor("#e0e0e0"), spaceAfter=8),

        Paragraph("3.1 LLM 选型建议", s["h2"]),
        Paragraph(
            "截至 2026 年，主流 LLM 提供商及其适用场景：", s["body"]
        ),
    ]

    # 表格
    table_data = [
        ["模型", "提供商", "特点", "适用场景"],
        ["claude-sonnet​-4-6", "Anthropic", "推理强、长上下文、中文优秀", "生产主力，复杂推理"],
        ["claude-haiku-4-5", "Anthropic", "速度快、成本低", "简单任务、大量调用"],
        ["gpt-4o", "OpenAI", "多模态、生态丰富", "代码生成、多模态"],
        ["gpt-4o-mini", "OpenAI", "低成本", "简单分类、摘要"],
        ["Qwen2.5-72B", "阿里云", "中文最强开源", "国内部署、私有化"],
        ["DeepSeek-V3", "DeepSeek", "高性价比", "成本敏感场景"],
    ]
    table = Table(table_data, colWidths=[3.5*cm, 2.5*cm, 5*cm, 4*cm])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#6C63FF")),
        ("TEXTCOLOR",  (0,0), (-1,0), colors.white),
        ("FONTNAME",   (0,0), (-1,-1), FONT_NAME),
        ("FONTSIZE",   (0,0), (-1,-1), 8),
        ("GRID",       (0,0), (-1,-1), 0.5, colors.HexColor("#cccccc")),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#f8f8ff")]),
        ("ALIGN",      (0,0), (-1,-1), "CENTER"),
        ("VALIGN",     (0,0), (-1,-1), "MIDDLE"),
        ("TOPPADDING", (0,0), (-1,-1), 5),
        ("BOTTOMPADDING", (0,0), (-1,-1), 5),
    ]))
    story += [table, Spacer(1, 0.4*cm)]

    story += [
        Paragraph("3.2 向量数据库选型", s["h2"]),
        Paragraph(
            "根据使用场景选择合适的向量数据库：", s["body"]
        ),
        Paragraph("• Chroma：纯 Python，零配置，适合本地开发和中小规模（<100万向量）。", s["bullet"]),
        Paragraph("• Pinecone：全托管云服务，适合快速上线、无运维负担场景。", s["bullet"]),
        Paragraph("• Milvus：高性能分布式，适合大规模（>1亿向量）生产场景。", s["bullet"]),
        Paragraph("• pgvector：PostgreSQL 扩展，适合已有 PG 基础设施的团队，减少技术栈复杂度。", s["bullet"]),
        Spacer(1, 0.3*cm),

        Paragraph("3.3 Embedding 模型选型", s["h2"]),
        Paragraph(
            "Embedding 质量直接影响检索效果。主流 Embedding 模型对比：", s["body"]
        ),
        Paragraph("• text-embedding-3-small（OpenAI）：1536维，性价比高，中英文均衡，推荐首选。", s["bullet"]),
        Paragraph("• text-embedding-3-large（OpenAI）：3072维，精度更高，成本约3倍。", s["bullet"]),
        Paragraph("• BGE-M3（智源研究院）：多语言、多粒度，中文表现优秀，支持本地部署。", s["bullet"]),
        Paragraph("• Cohere Embed v3：多语言支持强，适合国际化场景。", s["bullet"]),
        PageBreak(),
    ]

    # ── 第4章 产品规格 ────────────────────────────────────────────
    story += [
        Paragraph("第四章  产品规格与性能基准", s["h1"]),
        HRFlowable(width="100%", thickness=1, color=colors.HexColor("#e0e0e0"), spaceAfter=8),

        Paragraph("4.1 RAG 系统性能指标", s["h2"]),
        Paragraph(
            "以下为企业级 RAG 系统的生产环境性能基准（基于 claude-sonnet​-4-6 + Chroma + 混合检索）：", s["body"]
        ),
    ]
    perf_data = [
        ["指标", "目标值", "说明"],
        ["单次问答延迟", "≤ 5 秒", "P90 延迟，含检索+生成"],
        ["文档索引速度", "≤ 30 秒/100页", "PDF解析+分块+向量化"],
        ["检索准确率（MRR@4）", "≥ 0.75", "Top-4 召回中包含正确答案"],
        ["答案忠实度（Faithfulness）", "≥ 0.85", "回答基于检索内容，无幻觉"],
        ["并发支持", "50+ 用户", "同时发起问答请求"],
        ["知识库容量", "10万+ Chunks", "单实例 Chroma 支持"],
    ]
    perf_table = Table(perf_data, colWidths=[4.5*cm, 3.5*cm, 7*cm])
    perf_table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#2d2d6b")),
        ("TEXTCOLOR",  (0,0), (-1,0), colors.white),
        ("FONTNAME",   (0,0), (-1,-1), FONT_NAME),
        ("FONTSIZE",   (0,0), (-1,-1), 9),
        ("GRID",       (0,0), (-1,-1), 0.5, colors.HexColor("#cccccc")),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#f0f0ff")]),
        ("ALIGN",      (0,0), (-1,-1), "LEFT"),
        ("VALIGN",     (0,0), (-1,-1), "MIDDLE"),
        ("TOPPADDING", (0,0), (-1,-1), 6),
        ("BOTTOMPADDING", (0,0), (-1,-1), 6),
        ("LEFTPADDING",(0,0), (-1,-1), 8),
    ]))
    story += [perf_table, Spacer(1, 0.4*cm)]

    story += [
        Paragraph("4.2 产品定价与套餐", s["h2"]),
        Paragraph(
            "RAG 知识库产品采用 SaaS 订阅模式，提供三个套餐：", s["body"]
        ),
        Paragraph("• 基础版（¥299/月）：5个知识库，50MB存储，每月50万 Token，支持3用户。", s["bullet"]),
        Paragraph("• 专业版（¥999/月）：20个知识库，500MB存储，每月300万 Token，支持20用户，API访问。", s["bullet"]),
        Paragraph("• 企业版（¥4999/月）：无限知识库，10GB存储，每月2000万 Token，无限用户，私有部署选项。", s["bullet"]),

        Paragraph("4.3 安全与合规", s["h2"]),
        Paragraph(
            "企业级 RAG 系统需满足以下安全合规要求：", s["body"]
        ),
        Paragraph("• 数据隔离：多租户场景下，各用户知识库完全隔离，向量索引按租户分区。", s["bullet"]),
        Paragraph("• 传输加密：所有 API 通信使用 TLS 1.3 加密。", s["bullet"]),
        Paragraph("• 访问控制：基于 RBAC 的权限管理，文档级别的访问权限控制。", s["bullet"]),
        Paragraph("• 审计日志：记录所有查询请求、文档操作，日志保留90天。", s["bullet"]),
        Paragraph("• GDPR 合规：支持用户数据删除请求，数据在中国大陆服务器存储（信创合规）。", s["bullet"]),
        PageBreak(),
    ]

    # ── 第5章 实施路线图 ──────────────────────────────────────────
    story += [
        Paragraph("第五章  实施路线图与最佳实践", s["h1"]),
        HRFlowable(width="100%", thickness=1, color=colors.HexColor("#e0e0e0"), spaceAfter=8),

        Paragraph("5.1 企业 RAG 落地路线图", s["h2"]),
        Paragraph("Phase 1 — 概念验证（1-2周）：", s["body"]),
        Paragraph("选取 10-20 份代表性文档，搭建最小可用 RAG 系统，验证技术可行性和效果。", s["bullet"]),
        Paragraph("Phase 2 — 试点上线（2-4周）：", s["body"]),
        Paragraph("接入完整知识库，完善问答质量评估，对接企业 SSO，在特定团队试点。", s["bullet"]),
        Paragraph("Phase 3 — 规模推广（1-3月）：", s["body"]),
        Paragraph("全员推广，建立知识库运营体系，持续优化 Prompt 和检索策略，建立反馈闭环。", s["bullet"]),
        Spacer(1, 0.3*cm),

        Paragraph("5.2 常见问题与解决方案", s["h2"]),
        Paragraph(
            "问题1：回答不准确，引用了错误内容（幻觉问题）", s["body"]
        ),
        Paragraph("解决：降低生成温度（temperature=0），强化 System Prompt 的约束，增加 Faithfulness 评估。", s["bullet"]),
        Spacer(1, 0.1*cm),
        Paragraph(
            "问题2：检索召回率低，相关文档未被找到", s["body"]
        ),
        Paragraph("解决：引入混合检索（语义+BM25），调整 chunk_size，尝试 HyDE 或问题扩展技术。", s["bullet"]),
        Spacer(1, 0.1*cm),
        Paragraph(
            "问题3：多轮对话中代词理解错误", s["body"]
        ),
        Paragraph("解决：在检索前增加 Query Rewriting 步骤，将代词还原为具体实体再检索。", s["bullet"]),
        Spacer(1, 0.1*cm),
        Paragraph(
            "问题4：知识库更新后旧答案仍被召回", s["body"]
        ),
        Paragraph("解决：文档更新时重新生成对应 chunk 的向量（增量更新），避免全量重建。", s["bullet"]),

        Paragraph("5.3 评估体系", s["h2"]),
        Paragraph(
            "生产 RAG 系统必须建立完整的评估体系，核心指标包括：", s["body"]
        ),
        Paragraph("• Faithfulness（忠实度）：回答是否完全基于检索内容，无幻觉。", s["bullet"]),
        Paragraph("• Answer Relevancy（答案相关性）：回答是否切题，有无偏题。", s["bullet"]),
        Paragraph("• Context Precision（上下文精确率）：检索到的内容是否都有用。", s["bullet"]),
        Paragraph("• Context Recall（上下文召回率）：相关内容是否都被检索到。", s["bullet"]),
        Paragraph(
            "推荐使用 RAGAS 框架自动化评估，结合 LangSmith 追踪每次问答的完整链路。", s["body"]
        ),
        Spacer(1, 1*cm),

        HRFlowable(width="100%", thickness=1, color=colors.HexColor("#e0e0e0"), spaceAfter=8),
        Paragraph("© 2026 LangGraph-Claude 技术团队 · 本文档仅供内部学习使用", s["caption"]),
    ]

    doc.build(story)
    print(f"✅ 生成: {output_path}")


# ════════════════════════════════════════════════════════════════
# 文档 2：LangGraph 开发手册
# ════════════════════════════════════════════════════════════════
def build_langgraph_manual(output_path: str):
    doc = SimpleDocTemplate(
        output_path, pagesize=A4,
        leftMargin=2.5*cm, rightMargin=2.5*cm,
        topMargin=2.5*cm, bottomMargin=2.5*cm,
    )
    s = make_styles()
    story = []

    # ── 封面 ──────────────────────────────────────────────────────
    story += [
        Spacer(1, 3*cm),
        Paragraph("LangGraph 开发实战手册", s["title"]),
        Paragraph("基于图结构的 AI Agent 工作流编排指南", s["subtitle"]),
        Spacer(1, 0.5*cm),
        HRFlowable(width="80%", thickness=2, color=colors.HexColor("#FF6B35"), spaceAfter=20),
        Paragraph("版本 1.5 · 2026年4月", s["caption"]),
        Paragraph("适合人群：有 Python 基础的 AI 工程师", s["caption"]),
        PageBreak(),
    ]

    # ── 第1章 LangGraph 简介 ──────────────────────────────────────
    story += [
        Paragraph("第一章  LangGraph 核心概念", s["h1"]),
        HRFlowable(width="100%", thickness=1, color=colors.HexColor("#e0e0e0"), spaceAfter=8),

        Paragraph("1.1 为什么选择 LangGraph", s["h2"]),
        Paragraph(
            "LangGraph 是 LangChain 团队开发的 Agent 编排框架，基于有向图（Directed Graph）模型构建"
            "有状态的、可循环的 AI 工作流。相比传统的 Chain 模式，LangGraph 解决了以下痛点：", s["body"]
        ),
        Paragraph("• 支持循环（Loop）：Agent 可以反复执行某步骤（如重试、迭代优化），而 Chain 只能线性执行。", s["bullet"]),
        Paragraph("• 显式状态管理：所有 Agent 的状态统一在 State 对象中管理，清晰可追踪。", s["bullet"]),
        Paragraph("• 持久化支持：通过 Checkpointer 将状态保存到磁盘，支持断点续传和 Human-in-the-Loop。", s["bullet"]),
        Paragraph("• 可视化调试：与 LangSmith 深度集成，可视化每个节点的执行过程。", s["bullet"]),
        Spacer(1, 0.3*cm),

        Paragraph("1.2 核心概念：节点、边、状态", s["h2"]),
        Paragraph("LangGraph 的三个核心概念：", s["body"]),
        Paragraph(
            "Node（节点）：图中的执行单元，通常是一个 Python 函数或 Agent。"
            "每个节点接收当前状态（State）作为输入，返回状态的更新部分。", s["bullet"]
        ),
        Paragraph(
            "Edge（边）：连接节点的有向连线，分为普通边（固定路由）和条件边（根据状态动态路由）。"
            "条件边使得图可以根据执行结果动态选择下一步。", s["bullet"]
        ),
        Paragraph(
            "State（状态）：贯穿整个工作流的共享数据结构，通常用 TypedDict 定义。"
            "所有节点读取 State 并返回 State 的更新，StateGraph 自动合并更新。", s["bullet"]
        ),

        Paragraph("1.3 与 LangChain 的关系", s["h2"]),
        Paragraph(
            "LangGraph 是 LangChain 生态的一部分，完全兼容 LangChain 的 LLM、工具、Prompt 等组件。"
            "可以将 LangGraph 理解为 LangChain 的「编排层」：LangChain 提供基础组件（积木），"
            "LangGraph 提供组合方式（搭积木的规则）。两者配合使用，构成完整的 Agent 开发框架。", s["body"]
        ),
        PageBreak(),
    ]

    # ── 第2章 快速上手 ────────────────────────────────────────────
    story += [
        Paragraph("第二章  快速上手：Hello LangGraph", s["h1"]),
        HRFlowable(width="100%", thickness=1, color=colors.HexColor("#e0e0e0"), spaceAfter=8),

        Paragraph("2.1 环境安装", s["h2"]),
        Paragraph("通过 pip 安装 LangGraph 及相关依赖：", s["body"]),
        Paragraph(
            "pip install langgraph langchain-anthropic langchain-openai langsmith",
            s["code"]
        ),
        Spacer(1, 0.2*cm),

        Paragraph("2.2 第一个 LangGraph 程序", s["h2"]),
        Paragraph(
            "以下是一个最简单的 LangGraph Agent，展示核心概念：", s["body"]
        ),
        Paragraph(
            "from typing import TypedDict, Annotated\n"
            "from langgraph.graph import StateGraph, END\n"
            "from langgraph.graph.message import add_messages\n\n"
            "class State(TypedDict):\n"
            "    messages: Annotated[list, add_messages]\n\n"
            "def chatbot_node(state: State):\n"
            "    response = llm.invoke(state['messages'])\n"
            "    return {'messages': [response]}\n\n"
            "graph = StateGraph(State)\n"
            "graph.add_node('chatbot', chatbot_node)\n"
            "graph.set_entry_point('chatbot')\n"
            "graph.add_edge('chatbot', END)\n"
            "app = graph.compile()",
            s["code"]
        ),

        Paragraph("2.3 状态更新机制", s["h2"]),
        Paragraph(
            "LangGraph 使用 Reducer 机制合并状态更新。最常用的 Reducer 是 add_messages，"
            "它会将新消息追加到历史消息列表（而不是覆盖）。也可以自定义 Reducer：", s["body"]
        ),
        Paragraph(
            "# 自定义 Reducer：取两个整数中的最大值\n"
            "def max_reducer(a: int, b: int) -> int:\n"
            "    return max(a, b)\n\n"
            "class State(TypedDict):\n"
            "    count: Annotated[int, max_reducer]",
            s["code"]
        ),
        PageBreak(),
    ]

    # ── 第3章 高级特性 ────────────────────────────────────────────
    story += [
        Paragraph("第三章  高级特性", s["h1"]),
        HRFlowable(width="100%", thickness=1, color=colors.HexColor("#e0e0e0"), spaceAfter=8),

        Paragraph("3.1 条件路由（Conditional Edges）", s["h2"]),
        Paragraph(
            "条件路由是 LangGraph 的核心能力，允许根据当前状态动态决定下一步执行哪个节点：", s["body"]
        ),
        Paragraph(
            "def route_function(state: State) -> str:\n"
            "    last_msg = state['messages'][-1]\n"
            "    if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:\n"
            "        return 'tools'   # 有工具调用→执行工具\n"
            "    return 'end'         # 无工具调用→结束\n\n"
            "graph.add_conditional_edges(\n"
            "    'agent',\n"
            "    route_function,\n"
            "    {'tools': 'tool_node', 'end': END}\n"
            ")",
            s["code"]
        ),

        Paragraph("3.2 持久化与断点续传（Checkpointing）", s["h2"]),
        Paragraph(
            "LangGraph 的 Checkpointer 机制允许将每一步的状态保存到持久存储，"
            "实现以下能力：断点续传、Human-in-the-Loop、并发执行隔离。", s["body"]
        ),
        Paragraph(
            "from langgraph.checkpoint.sqlite import SqliteSaver\n\n"
            "# 使用 SQLite 持久化\n"
            "checkpointer = SqliteSaver.from_conn_string('state.db')\n"
            "app = graph.compile(checkpointer=checkpointer)\n\n"
            "# 每个会话用独立的 thread_id\n"
            "config = {'configurable': {'thread_id': 'session_001'}}\n"
            "result = app.invoke(initial_state, config=config)",
            s["code"]
        ),

        Paragraph("3.3 Human-in-the-Loop（interrupt）", s["h2"]),
        Paragraph(
            "在需要人工审核的场景，使用 interrupt() 暂停 Graph 执行，等待外部输入：", s["body"]
        ),
        Paragraph(
            "from langgraph.types import interrupt, Command\n\n"
            "def review_node(state: State):\n"
            "    # 暂停执行，等待人工决策\n"
            "    decision = interrupt({'data': state['draft']})\n"
            "    return {'human_decision': decision}\n\n"
            "# 恢复执行（携带人工决策）\n"
            "app.invoke(Command(resume='approve'), config=config)",
            s["code"]
        ),

        Paragraph("3.4 多 Agent 模式", s["h2"]),
        Paragraph(
            "LangGraph 支持多种多 Agent 架构。Supervisor 模式是最常用的：", s["body"]
        ),
        Paragraph("• Supervisor Agent：监督者，负责任务分发和结果聚合。", s["bullet"]),
        Paragraph("• Worker Agents：专业化子 Agent，每个专注于特定领域（搜索、写作、审校等）。", s["bullet"]),
        Paragraph("• 消息传递：通过 State 中的 messages 字段传递上下文。", s["bullet"]),
        PageBreak(),
    ]

    # ── 第4章 生产部署 ────────────────────────────────────────────
    story += [
        Paragraph("第四章  生产部署最佳实践", s["h1"]),
        HRFlowable(width="100%", thickness=1, color=colors.HexColor("#e0e0e0"), spaceAfter=8),

        Paragraph("4.1 LangSmith 可观测性", s["h2"]),
        Paragraph(
            "在生产环境中，必须配置 LangSmith 追踪每次 Agent 执行的完整链路：", s["body"]
        ),
        Paragraph(
            "import os\n"
            "os.environ['LANGCHAIN_TRACING_V2'] = 'true'\n"
            "os.environ['LANGCHAIN_PROJECT'] = 'my-agent-project'\n"
            "os.environ['LANGCHAIN_API_KEY'] = 'your-langsmith-key'",
            s["code"]
        ),
        Paragraph(
            "配置后，每次 Graph 执行的完整链路（节点输入/输出、Token 消耗、耗时）"
            "都会上报到 LangSmith 平台，可在 Web UI 中可视化查看和调试。", s["body"]
        ),

        Paragraph("4.2 错误处理与重试", s["h2"]),
        Paragraph(
            "生产 Agent 必须具备完善的错误处理机制：", s["body"]
        ),
        Paragraph("• API 超时：设置合理的超时时间（30s），超时自动重试。", s["bullet"]),
        Paragraph("• 速率限制：捕获 RateLimitError，使用指数退避重试（最多3次）。", s["bullet"]),
        Paragraph("• 模型降级：主模型不可用时，自动切换到备用模型。", s["bullet"]),
        Paragraph("• 最大迭代数：设置 Agent 最大步骤数（如10步），防止无限循环。", s["bullet"]),

        Paragraph("4.3 成本控制", s["h2"]),
        Paragraph("有效控制 LLM API 成本的策略：", s["body"]),
        Paragraph("• 模型分级：复杂推理用 claude-sonnet，简单任务用 claude-haiku，降低平均成本。", s["bullet"]),
        Paragraph("• 缓存：对相同输入启用语义缓存（LangChain InMemoryCache / Redis Cache）。", s["bullet"]),
        Paragraph("• Token 限制：设置 max_tokens 上限，防止意外生成大量文本。", s["bullet"]),
        Paragraph("• 预算告警：在 LangSmith 设置 Token 预算，超出时触发告警。", s["bullet"]),

        Paragraph("4.4 LangGraph 部署选项", s["h2"]),
        Paragraph("LangGraph 支持多种部署模式：", s["body"]),
        Paragraph("• 本地部署：直接在服务器运行，适合内网场景。", s["bullet"]),
        Paragraph("• LangGraph Platform：官方托管服务，提供 API Server、监控、扩缩容能力。", s["bullet"]),
        Paragraph("• Docker 容器化：打包为 Docker 镜像，部署到 Kubernetes 集群。", s["bullet"]),
        Spacer(1, 1*cm),

        HRFlowable(width="100%", thickness=1, color=colors.HexColor("#e0e0e0"), spaceAfter=8),
        Paragraph("© 2026 LangGraph-Claude 技术团队 · 本文档仅供内部学习使用", s["caption"]),
    ]

    doc.build(story)
    print(f"✅ 生成: {output_path}")


if __name__ == "__main__":
    import sys
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "./data/documents"
    os.makedirs(out_dir, exist_ok=True)
    build_ai_agent_whitepaper(f"{out_dir}/AI_Agent_技术白皮书.pdf")
    build_langgraph_manual(f"{out_dir}/LangGraph_开发手册.pdf")
    print("\\n✅ 所有示例 PDF 生成完成！")
