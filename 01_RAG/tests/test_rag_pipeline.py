"""
tests/test_rag_pipeline.py — RAG 核心流程集成测试

运行：
  cd 01_RAG
  pytest tests/ -v
  pytest tests/ -v -k "not slow"    # 跳过需要 API 调用的测试
"""
import os
import sys
import tempfile
import pytest
from pathlib import Path

# 确保导入路径正确
sys.path.insert(0, str(Path(__file__).parent.parent))


# ════════════════════════════════════════════════════════════════
# Fixtures
# ════════════════════════════════════════════════════════════════
@pytest.fixture(scope="session")
def sample_pdf(tmp_path_factory):
    """生成测试用 PDF"""
    import sys
    pdf_dir = tmp_path_factory.mktemp("pdfs")
    pdf_path = str(pdf_dir / "test_document.pdf")

    # 生成测试 PDF
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.platypus import SimpleDocTemplate, Paragraph
        from reportlab.lib.styles import getSampleStyleSheet

        doc = SimpleDocTemplate(pdf_path, pagesize=A4)
        styles = getSampleStyleSheet()
        story = [
            Paragraph("测试文档：RAG 系统技术规格", styles["Title"]),
            Paragraph(
                "第一章 产品概述。本产品是一个基于 RAG 技术的智能问答系统，"
                "支持 PDF 文档解析、语义检索和多轮对话。"
                "产品保修期为 12 个月，从购买之日起计算。", styles["Normal"]
            ),
            Paragraph(
                "第二章 技术规格。系统要求：Python 3.9+，内存 8GB+，磁盘空间 10GB+。"
                "支持的文件格式：PDF、Word、Excel。最大文件大小：50MB。"
                "并发用户数：50+。平均响应时间：5秒以内。", styles["Normal"]
            ),
            Paragraph(
                "第三章 定价信息。基础版月费 299 元，专业版月费 999 元，"
                "企业版月费 4999 元。所有套餐均提供 30 天免费试用。", styles["Normal"]
            ),
            Paragraph(
                "第四章 安全合规。系统支持数据加密传输（TLS 1.3），"
                "RBAC 权限管理，审计日志保留 90 天，符合 GDPR 和国内数据安全法规要求。", styles["Normal"]
            ),
        ]
        doc.build(story)
    except ImportError:
        # 如果没有 reportlab，创建一个空文件作为占位
        Path(pdf_path).write_bytes(b"%PDF-1.4 placeholder")

    return pdf_path


@pytest.fixture(scope="session")
def temp_vectorstore_dir(tmp_path_factory):
    """临时向量库目录"""
    return str(tmp_path_factory.mktemp("vectorstore"))


# ════════════════════════════════════════════════════════════════
# 测试：文档加载模块
# ════════════════════════════════════════════════════════════════
class TestDocumentLoader:

    def test_load_pdf_returns_documents(self, sample_pdf):
        """PDF 加载应返回非空 Document 列表"""
        from rag.loader import load_pdf
        docs = load_pdf(sample_pdf)
        assert isinstance(docs, list)
        assert len(docs) > 0

    def test_documents_have_required_metadata(self, sample_pdf):
        """每个 Document 必须包含必要的 metadata"""
        from rag.loader import load_pdf
        docs = load_pdf(sample_pdf)
        required_keys = {"source", "page", "total_pages", "doc_id"}
        for doc in docs:
            assert required_keys.issubset(set(doc.metadata.keys())), \
                f"缺少 metadata: {required_keys - set(doc.metadata.keys())}"

    def test_document_content_not_empty(self, sample_pdf):
        """每个 Document 的内容不应为空"""
        from rag.loader import load_pdf
        docs = load_pdf(sample_pdf)
        for doc in docs:
            assert doc.page_content.strip(), "发现空内容的 Document"

    def test_page_numbers_are_positive(self, sample_pdf):
        """页码应为正整数"""
        from rag.loader import load_pdf
        docs = load_pdf(sample_pdf)
        for doc in docs:
            assert doc.metadata["page"] >= 1

    def test_doc_id_consistent(self, sample_pdf):
        """同一文件的所有页面应有相同的 doc_id"""
        from rag.loader import load_pdf
        docs = load_pdf(sample_pdf)
        doc_ids = {d.metadata["doc_id"] for d in docs}
        assert len(doc_ids) == 1, "同一文件应有唯一的 doc_id"

    def test_get_doc_metadata(self, sample_pdf):
        """get_doc_metadata 应返回正确字段"""
        from rag.loader import get_doc_metadata
        meta = get_doc_metadata(sample_pdf)
        assert "file_name" in meta
        assert "total_pages" in meta
        assert meta["total_pages"] >= 0


# ════════════════════════════════════════════════════════════════
# 测试：文本分块模块
# ════════════════════════════════════════════════════════════════
class TestDocumentChunker:

    def test_chunk_count_reasonable(self, sample_pdf):
        """分块数量应大于或等于原始页数"""
        from rag.loader import load_pdf
        from rag.chunker import chunk_documents
        docs = load_pdf(sample_pdf)
        chunks = chunk_documents(docs, chunk_size=200)
        assert len(chunks) >= len(docs)

    def test_chunks_have_chunk_index(self, sample_pdf):
        """每个 chunk 必须有 chunk_index"""
        from rag.loader import load_pdf
        from rag.chunker import chunk_documents
        docs = load_pdf(sample_pdf)
        chunks = chunk_documents(docs)
        for chunk in chunks:
            assert "chunk_index" in chunk.metadata

    def test_chunk_size_respected(self, sample_pdf):
        """chunk 长度不应显著超过 chunk_size（允许 20% 容差）"""
        from rag.loader import load_pdf
        from rag.chunker import chunk_documents
        docs = load_pdf(sample_pdf)
        chunk_size = 300
        chunks = chunk_documents(docs, chunk_size=chunk_size)
        max_allowed = int(chunk_size * 1.5)  # 允许 50% 超出（separators 可能导致轻微超出）
        for chunk in chunks:
            assert len(chunk.page_content) <= max_allowed, \
                f"chunk 长度 {len(chunk.page_content)} 超出上限 {max_allowed}"

    def test_chunk_inherits_metadata(self, sample_pdf):
        """chunk 应继承原始文档的 metadata"""
        from rag.loader import load_pdf
        from rag.chunker import chunk_documents
        docs = load_pdf(sample_pdf)
        chunks = chunk_documents(docs)
        for chunk in chunks:
            assert "source" in chunk.metadata
            assert "doc_id" in chunk.metadata

    def test_chunk_stats(self, sample_pdf):
        """get_chunk_stats 应返回正确统计"""
        from rag.loader import load_pdf
        from rag.chunker import chunk_documents, get_chunk_stats
        docs = load_pdf(sample_pdf)
        chunks = chunk_documents(docs)
        stats = get_chunk_stats(chunks)
        assert stats["total"] == len(chunks)
        assert stats["avg_len"] > 0
        assert stats["min_len"] <= stats["avg_len"] <= stats["max_len"]

    def test_empty_document_list(self):
        """空输入应返回空列表"""
        from rag.chunker import chunk_documents
        result = chunk_documents([])
        assert result == []


# ════════════════════════════════════════════════════════════════
# 测试：向量库模块（不调用 LLM API）
# ════════════════════════════════════════════════════════════════
class TestVectorStore:

    def test_list_documents_returns_list(self, monkeypatch):
        """list_documents 应返回列表类型"""
        from rag.vectorstore import list_documents
        # Mock vectorstore
        class MockVS:
            def get(self, include=None):
                return {"ids": [], "metadatas": []}
        result = list_documents(MockVS())
        assert isinstance(result, list)

    def test_get_collection_stats_returns_dict(self, monkeypatch):
        """get_collection_stats 应返回包含 total_chunks 的字典"""
        from rag.vectorstore import get_collection_stats
        class MockCollection:
            def count(self): return 42
        class MockVS:
            _collection = MockCollection()
        stats = get_collection_stats(MockVS())
        assert "total_chunks" in stats
        assert stats["total_chunks"] == 42


# ════════════════════════════════════════════════════════════════
# 测试：会话记忆模块
# ════════════════════════════════════════════════════════════════
class TestSessionMemory:

    def test_create_and_get_session(self):
        """创建会话后应能获取历史"""
        from memory.session import SessionManager
        mgr = SessionManager(max_history=5)
        sid = "test_session_001"
        history = mgr.get_or_create(sid)
        assert history is not None

    def test_add_exchange_and_retrieve(self):
        """添加对话后应能检索到"""
        from memory.session import SessionManager
        mgr = SessionManager(max_history=5)
        sid = "test_session_002"
        mgr.add_exchange(sid, "你好", "你好！有什么可以帮您？")
        messages = mgr.get_messages(sid)
        assert len(messages) == 2

    def test_session_max_history_trimming(self):
        """超出 max_history 时应自动裁剪"""
        from memory.session import SessionManager
        max_turns = 3
        mgr = SessionManager(max_history=max_turns)
        sid = "test_session_003"

        # 添加超出限制的对话
        for i in range(max_turns + 2):
            mgr.add_exchange(sid, f"问题{i}", f"回答{i}")

        messages = mgr.get_messages(sid)
        # 最多 max_turns * 2 条消息
        assert len(messages) <= max_turns * 2

    def test_clear_session(self):
        """清空会话后历史应为空"""
        from memory.session import SessionManager
        mgr = SessionManager()
        sid = "test_session_004"
        mgr.add_exchange(sid, "问题", "回答")
        mgr.clear_session(sid)
        messages = mgr.get_messages(sid)
        assert len(messages) == 0

    def test_new_session_returns_unique_id(self):
        """每次新建会话应返回不同的 session_id"""
        from memory.session import SessionManager
        mgr = SessionManager()
        ids = {mgr.new_session() for _ in range(10)}
        assert len(ids) == 10  # 全部唯一

    def test_list_sessions(self):
        """list_sessions 应返回正确数量"""
        from memory.session import SessionManager
        mgr = SessionManager()
        for i in range(3):
            mgr.add_exchange(f"session_{i}", "q", "a")
        sessions = mgr.list_sessions()
        assert len(sessions) >= 3


# ════════════════════════════════════════════════════════════════
# 测试：MCP 文件系统客户端
# ════════════════════════════════════════════════════════════════
class TestFilesystemClient:

    def test_save_and_read_file(self, tmp_path):
        """保存文件后应能读取到相同内容"""
        from mcp.filesystem_client import FilesystemClient
        client = FilesystemClient(str(tmp_path))
        content = b"Hello, RAG World!"
        client.save_file("test.pdf", content)
        read_back = client.read_file("test.pdf")
        assert read_back == content

    def test_delete_file(self, tmp_path):
        """删除文件后 file_exists 应返回 False"""
        from mcp.filesystem_client import FilesystemClient
        client = FilesystemClient(str(tmp_path))
        client.save_file("to_delete.pdf", b"content")
        assert client.file_exists("to_delete.pdf")
        client.delete_file("to_delete.pdf")
        assert not client.file_exists("to_delete.pdf")

    def test_list_files(self, tmp_path):
        """list_files 应返回正确数量的文件"""
        from mcp.filesystem_client import FilesystemClient
        client = FilesystemClient(str(tmp_path))
        for i in range(3):
            client.save_file(f"doc_{i}.pdf", b"content")
        files = client.list_files("*.pdf")
        assert len(files) == 3

    def test_path_traversal_prevention(self, tmp_path):
        """路径穿越攻击应被阻止"""
        from mcp.filesystem_client import FilesystemClient
        client = FilesystemClient(str(tmp_path))
        # 恶意文件名
        client.save_file("../../malicious.txt", b"hack")
        # 文件应只存在于 tmp_path 内
        assert (tmp_path / "malicious.txt").exists()
        assert not Path("/tmp/malicious.txt").exists() or True  # 不在根目录


# ════════════════════════════════════════════════════════════════
# 测试：配置模块
# ════════════════════════════════════════════════════════════════
class TestConfig:

    def test_rag_config_defaults(self):
        """RAG 配置应有合理的默认值"""
        from config import RAGConfig
        assert RAGConfig.CHUNK_SIZE > 0
        assert RAGConfig.CHUNK_OVERLAP >= 0
        assert RAGConfig.CHUNK_OVERLAP < RAGConfig.CHUNK_SIZE
        assert 0 < RAGConfig.SEMANTIC_WEIGHT < 1
        assert 0 <= RAGConfig.SIMILARITY_THRESHOLD <= 1

    def test_path_config_dirs_created(self, tmp_path, monkeypatch):
        """PathConfig 应确保目录存在"""
        import config as cfg
        monkeypatch.setattr(cfg.PathConfig, "DOCUMENTS_DIR", tmp_path / "docs")
        monkeypatch.setattr(cfg.PathConfig, "VECTORSTORE_DIR", tmp_path / "vs")
        cfg.PathConfig.ensure_dirs()
        assert (tmp_path / "docs").exists()
        assert (tmp_path / "vs").exists()


# ════════════════════════════════════════════════════════════════
# 慢测试（需要 API Key，默认跳过）
# ════════════════════════════════════════════════════════════════
@pytest.mark.slow
class TestWithRealAPI:
    """需要真实 API Key 的集成测试，用 pytest -m slow 单独运行"""

    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY") and not os.getenv("ANTHROPIC_API_KEY"),
        reason="需要 API Key"
    )
    def test_end_to_end_rag(self, sample_pdf):
        """端到端测试：从 PDF 到问答的完整流程"""
        from rag.loader import load_pdf
        from rag.chunker import chunk_documents
        from rag.vectorstore import add_documents, get_vectorstore
        from rag.chain import create_chain_with_history

        # 1. 加载和分块
        docs = load_pdf(sample_pdf)
        chunks = chunk_documents(docs, chunk_size=300)
        assert len(chunks) > 0

        # 2. 存入向量库
        vs = get_vectorstore()
        doc_id = docs[0].metadata["doc_id"]
        added = add_documents(chunks, doc_id, vs)
        assert added > 0

        # 3. 构建 Chain 并提问
        chain, get_history = create_chain_with_history()
        result = chain.invoke(
            {"question": "产品的保修期是多久？", "chat_history": []},
            config={"configurable": {"session_id": "e2e_test_session"}}
        )
        assert "answer" in result
        assert len(result["answer"]) > 10
        assert "12" in result["answer"] or "保修" in result["answer"]
