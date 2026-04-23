"""
app.py — RAG 知识库问答系统主应用
Streamlit 生产级 UI
"""
from __future__ import annotations
import uuid
import time
import sys
import os
from pathlib import Path

# 确保项目根目录在 sys.path
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st

# ── 页面配置（必须是第一个 Streamlit 调用） ──────────────────────
st.set_page_config(
    page_title="智能知识库问答",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── 延迟导入（避免启动时崩溃） ───────────────────────────────────
from config import llm_config, rag_config, path_config
from mcp_local.filesystem_client import get_filesystem_client
from rag.loader import load_pdf, get_doc_metadata
from rag.chunker import chunk_documents
from rag.vectorstore import (
    get_vectorstore, add_documents, delete_document,
    list_documents, get_collection_stats
)
from rag.chain import create_chain_with_history, format_docs_for_context
from memory.session import get_session_manager


# ════════════════════════════════════════════════════════════════
# CSS 样式注入
# ════════════════════════════════════════════════════════════════
def inject_css():
    st.markdown("""
    <style>
    /* 消息气泡 */
    .user-bubble {
        background: linear-gradient(135deg, #6C63FF, #4a42cc);
        color: white;
        padding: 12px 16px;
        border-radius: 18px 18px 4px 18px;
        margin: 8px 0;
        max-width: 80%;
        margin-left: auto;
        box-shadow: 0 2px 8px rgba(108,99,255,0.3);
    }
    .ai-bubble {
        background: #1E2130;
        color: #FAFAFA;
        padding: 12px 16px;
        border-radius: 18px 18px 18px 4px;
        margin: 8px 0;
        max-width: 85%;
        border-left: 3px solid #6C63FF;
        box-shadow: 0 2px 8px rgba(0,0,0,0.3);
    }
    /* 来源标签 */
    .source-tag {
        display: inline-block;
        background: rgba(108,99,255,0.15);
        color: #a09fff;
        border: 1px solid rgba(108,99,255,0.3);
        border-radius: 12px;
        padding: 2px 10px;
        font-size: 0.75rem;
        margin: 2px 3px;
        cursor: default;
    }
    /* 指标卡片 */
    .metric-card {
        background: #1E2130;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        border: 1px solid rgba(108,99,255,0.2);
    }
    /* 文档列表 */
    .doc-row {
        background: #1E2130;
        border-radius: 8px;
        padding: 10px 14px;
        margin: 6px 0;
        border-left: 3px solid #6C63FF;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    /* 思考动画 */
    .thinking-dots::after {
        content: '...';
        animation: dots 1.5s steps(4, end) infinite;
    }
    @keyframes dots {
        0%, 20% { content: ''; }
        40% { content: '.'; }
        60% { content: '..'; }
        80%, 100% { content: '...'; }
    }
    /* 去除 Streamlit 默认 padding */
    .block-container { padding-top: 1rem; }
    </style>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
# 初始化状态
# ════════════════════════════════════════════════════════════════
def init_session_state():
    defaults = {
        "session_id": str(uuid.uuid4())[:8],
        "chat_history": [],       # [{role, content, sources, time_ms}]
        "chain": None,
        "chain_get_history": None,
        "vectorstore": None,
        "indexed_docs": [],       # 已索引文档列表
        "is_indexing": False,
        "retriever": None,
        # 配置
        "top_k": rag_config.FINAL_TOP_K,
        "threshold": rag_config.SIMILARITY_THRESHOLD,
        "show_sources": True,
        "show_debug": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def get_or_build_chain():
    """懒初始化 RAG Chain"""
    if st.session_state.chain is None:
        try:
            chain, get_history = create_chain_with_history()
            st.session_state.chain = chain
            st.session_state.chain_get_history = get_history
        except EnvironmentError as e:
            st.error(f"❌ {e}")
            return None, None
    return st.session_state.chain, st.session_state.chain_get_history


def refresh_indexed_docs():
    """刷新已索引文档列表"""
    try:
        vs = get_vectorstore()
        st.session_state.indexed_docs = list_documents(vs)
    except Exception:
        st.session_state.indexed_docs = []


# ════════════════════════════════════════════════════════════════
# 侧边栏
# ════════════════════════════════════════════════════════════════
def render_sidebar():
    with st.sidebar:
        # Logo
        st.markdown("""
        <div style="text-align:center; padding:10px 0 20px;">
            <div style="font-size:2.5rem;">🧠</div>
            <div style="font-size:1.2rem; font-weight:700; color:#6C63FF;">智能知识库</div>
            <div style="font-size:0.75rem; color:#888; margin-top:4px;">RAG Knowledge Base v2.0</div>
        </div>
        """, unsafe_allow_html=True)

        # ── 文档管理 ──────────────────────────────────────────────
        st.subheader("📁 文档管理")

        uploaded = st.file_uploader(
            "上传 PDF 文档",
            type=["pdf"],
            accept_multiple_files=True,
            help="支持多文件同时上传，单文件 ≤ 50MB",
        )

        if uploaded:
            if st.button("🚀 开始索引", type="primary", use_container_width=True):
                _handle_document_upload(uploaded)

        # 已索引文档列表
        if st.session_state.indexed_docs:
            st.markdown("**已索引文档**")
            for doc in st.session_state.indexed_docs:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(
                        f"📄 **{doc['source']}**  \n"
                        f"<span style='color:#888;font-size:0.75rem;'>"
                        f"{doc.get('child_count', doc['total_chunks'])} 子块 · "
                        f"{doc.get('parent_count', 0)} 父块 · "
                        f"{doc['total_pages']} 页</span>",
                        unsafe_allow_html=True
                    )
                with col2:
                    if st.button("🗑️", key=f"del_{doc['doc_id']}", help="删除此文档"):
                        _handle_document_delete(doc["doc_id"], doc["source"])
        else:
            st.info("暂无已索引文档\n\n请上传 PDF 文件开始使用")

        # ── 检索配置 ──────────────────────────────────────────────
        st.divider()
        st.subheader("⚙️ 检索配置")

        st.session_state.top_k = st.slider(
            "Top-K 召回数量", 1, 10, st.session_state.top_k,
            help="每次检索返回的最相关文档块数量"
        )
        st.session_state.threshold = st.slider(
            "相似度阈值", 0.0, 1.0, st.session_state.threshold, 0.05,
            help="低于此阈值的检索结果将被过滤"
        )
        st.session_state.show_sources = st.toggle(
            "显示来源", st.session_state.show_sources
        )
        st.session_state.show_debug = st.toggle(
            "调试模式", st.session_state.show_debug,
            help="显示检索到的原始文档块"
        )

        # ── 会话管理 ──────────────────────────────────────────────
        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 新对话", use_container_width=True):
                _new_session()
        with col2:
            if st.button("🗑️ 清空记录", use_container_width=True):
                st.session_state.chat_history = []
                session_mgr = get_session_manager()
                session_mgr.clear_session(st.session_state.session_id)
                st.rerun()

        # ── 统计信息 ──────────────────────────────────────────────
        st.divider()
        st.caption("📊 系统状态")
        stats = get_collection_stats()
        st.caption(
            f"索引: {stats.get('total_children', 0)} 个子块 / "
            f"{stats.get('total_parents', 0)} 个父块"
        )
        st.caption(f"对话轮次: {len(st.session_state.chat_history)}")
        st.caption(f"会话 ID: {st.session_state.session_id}")


def _handle_document_upload(uploaded_files):
    """处理文档上传与索引"""
    fs_client = get_filesystem_client(str(path_config.DOCUMENTS_DIR))
    vs = get_vectorstore()

    progress = st.sidebar.progress(0, text="初始化中...")
    total = len(uploaded_files)

    for i, uploaded_file in enumerate(uploaded_files):
        # 1. 保存文件
        progress.progress((i / total) * 0.1, text=f"保存 {uploaded_file.name}...")
        file_bytes = uploaded_file.read()
        saved_path = fs_client.save_file(uploaded_file.name, file_bytes)

        # 2. 解析
        progress.progress((i / total) * 0.3 + 0.1, text=f"解析 {uploaded_file.name}...")
        try:
            pages = load_pdf(saved_path)
        except Exception as e:
            st.sidebar.error(f"❌ 解析失败: {uploaded_file.name} - {e}")
            continue

        # 3. 分块
        progress.progress((i / total) * 0.5 + 0.3, text=f"分块 {uploaded_file.name}...")
        chunks = chunk_documents(pages)

        # 4. 向量化并存入 ChromaDB
        progress.progress((i / total) * 0.9 + 0.1, text=f"向量化 {uploaded_file.name}...")
        doc_id = pages[0].metadata["doc_id"] if pages else uploaded_file.name
        added = add_documents(chunks, doc_id, vs)

        # 5. 重建检索器（包含新文档）
        st.session_state.retriever = None  # 强制重建

        st.sidebar.success(f"✅ {uploaded_file.name}: {added} 块已索引")

    progress.progress(1.0, text="完成！")
    time.sleep(0.5)
    progress.empty()

    # 刷新文档列表，重置 chain（含新文档）
    refresh_indexed_docs()
    st.session_state.chain = None  # 强制重建 chain
    st.rerun()


def _handle_document_delete(doc_id: str, source: str):
    """处理文档删除"""
    vs = get_vectorstore()
    deleted = delete_document(doc_id, vs)
    if deleted > 0:
        # 同时删除原始文件
        fs_client = get_filesystem_client(str(path_config.DOCUMENTS_DIR))
        fs_client.delete_file(source)
        st.session_state.chain = None
        refresh_indexed_docs()
        st.rerun()
    else:
        st.sidebar.error("删除失败")


def _new_session():
    """开始新对话"""
    session_mgr = get_session_manager()
    new_id = session_mgr.new_session()
    st.session_state.session_id = new_id
    st.session_state.chat_history = []
    st.rerun()


# ════════════════════════════════════════════════════════════════
# 主聊天区域
# ════════════════════════════════════════════════════════════════
def render_chat_area():
    st.markdown("""
    <div style="text-align:center; padding: 10px 0 20px;">
        <h2 style="color:#6C63FF; margin:0;">🧠 智能知识库问答</h2>
        <p style="color:#888; margin:4px 0 0;">基于 RAG · 混合检索 · 多轮对话 · 来源可追溯</p>
    </div>
    """, unsafe_allow_html=True)

    # 空知识库提示
    if not st.session_state.indexed_docs:
        st.info(
            "📂 **知识库为空**  \n\n"
            "请在左侧侧边栏上传 PDF 文档，系统将自动完成解析、分块和向量化。  \n\n"
            "💡 **快速开始**：点击左侧上传区域，选择您的 PDF 文件即可开始。",
            icon="💡"
        )
        return

    # 渲染历史消息
    chat_container = st.container()
    with chat_container:
        for turn in st.session_state.chat_history:
            _render_message(turn)

    # 输入框（固定在底部）
    st.divider()
    _render_input_area()


def _render_message(turn: dict):
    """渲染单条消息"""
    role = turn["role"]
    content = turn["content"]

    if role == "user":
        with st.chat_message("user", avatar="👤"):
            st.markdown(content)
    else:
        with st.chat_message("assistant", avatar="🧠"):
            st.markdown(content)

            # 来源展示
            if st.session_state.show_sources and turn.get("sources"):
                sources = turn["sources"]
                source_tags = " ".join([
                    f'<span class="source-tag">📄 {s["source"]} · 第{s["page"]}页</span>'
                    for s in sources
                ])
                st.markdown(
                    f"<div style='margin-top:8px;'>{source_tags}</div>",
                    unsafe_allow_html=True
                )

            # 调试：显示检索到的原始文档块
            if st.session_state.show_debug and turn.get("raw_docs"):
                with st.expander("🔍 检索到的原始文档块"):
                    for i, doc in enumerate(turn["raw_docs"], 1):
                        score = doc.get("similarity_score", "N/A")
                        st.markdown(f"**[块 {i}]** {doc['source']} · 第{doc['page']}页 · 相似度:{score}")
                        st.text(doc["content"][:300] + "..." if len(doc["content"]) > 300 else doc["content"])
                        st.divider()

            # 响应时间
            if turn.get("time_ms"):
                st.caption(f"⏱️ {turn['time_ms']}ms")


def _render_input_area():
    """渲染消息输入区域"""
    col1, col2 = st.columns([6, 1])

    with col1:
        user_input = st.chat_input(
            "输入您的问题（支持多轮对话）...",
            key="user_input"
        )

    if user_input and user_input.strip():
        _process_query(user_input.strip())


def _process_query(query: str):
    """处理用户查询"""
    # 立即显示用户消息
    st.session_state.chat_history.append({
        "role": "user",
        "content": query,
    })

    # 构建 Chain
    chain, get_history = get_or_build_chain()
    if chain is None:
        st.error("❌ 请先在 .env 文件中配置 API Key")
        return

    # 调用 RAG Chain
    with st.spinner("🔍 检索知识库中..."):
        t0 = time.time()
        try:
            result = chain.invoke(
                {
                    "question": query,
                    "chat_history": [],  # 由 RunnableWithMessageHistory 自动注入
                },
                config={"configurable": {"session_id": st.session_state.session_id}},
            )
            elapsed_ms = round((time.time() - t0) * 1000)

            answer = result.get("answer", "抱歉，生成回答时出现问题。")
            source_docs = result.get("sources", [])

            # 整理来源信息
            sources = []
            raw_docs = []
            seen_sources = set()
            for doc in source_docs:
                source = doc.metadata.get("source", "未知")
                page = doc.metadata.get("page", "?")
                key = f"{source}_{page}"
                if key not in seen_sources:
                    seen_sources.add(key)
                    sources.append({"source": source, "page": page})
                raw_docs.append({
                    "source": source,
                    "page": page,
                    "content": doc.page_content,
                    "similarity_score": doc.metadata.get("similarity_score", "N/A"),
                })

            st.session_state.chat_history.append({
                "role": "assistant",
                "content": answer,
                "sources": sources,
                "raw_docs": raw_docs,
                "time_ms": elapsed_ms,
            })

        except Exception as e:
            err_msg = f"❌ 处理失败：{str(e)}"
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": err_msg,
                "sources": [],
            })

    st.rerun()


# ════════════════════════════════════════════════════════════════
# API Key 检测页
# ════════════════════════════════════════════════════════════════
def render_api_key_setup():
    st.warning("⚠️ 未检测到有效的 API Key，请先配置后使用。")

    with st.expander("📋 配置指引", expanded=True):
        st.markdown("""
        **第一步：复制环境变量模板**
        ```bash
        cp .env.example .env
        ```

        **第二步：编辑 `.env` 文件，填入您的 API Key**
        ```bash
        # 选择一个 LLM 提供商：
        ANTHROPIC_API_KEY=sk-ant-xxxx   # Claude (推荐)
        OPENAI_API_KEY=sk-xxxx           # GPT-4o

        # OpenAI Embedding（用于向量化，需要 OpenAI Key）
        # 若无 OpenAI Key，系统将自动使用本地 HuggingFace 模型
        ```

        **第三步：重启应用**
        ```bash
        streamlit run app.py
        ```
        """)


# ════════════════════════════════════════════════════════════════
# 主入口
# ════════════════════════════════════════════════════════════════
def main():
    inject_css()
    init_session_state()

    # 检查 API Key
    try:
        llm_config.validate()
    except EnvironmentError:
        render_api_key_setup()
        # 即使没有 API Key 也允许查看界面
        render_sidebar()
        return

    # 初次加载时刷新文档列表
    if not st.session_state.get("_initialized"):
        refresh_indexed_docs()
        st.session_state["_initialized"] = True

    # 渲染
    render_sidebar()
    render_chat_area()


if __name__ == "__main__":
    main()
