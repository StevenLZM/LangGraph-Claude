"""
config.py — 全局配置中心
从环境变量加载，提供合理默认值，所有模块均从此处导入配置

支持的 LLM 提供商（优先级从高到低）：
  1. DeepSeek              — DEEPSEEK_API_KEY
  2. DashScope (千问/Qwen) — DASHSCOPE_API_KEY
  3. Anthropic (Claude)    — ANTHROPIC_API_KEY
  4. OpenAI (GPT)          — OPENAI_API_KEY
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# 加载 .env 文件（优先级低于系统环境变量）
load_dotenv()


def _consume_legacy_milvus_uri() -> str:
    """
    Backwards-compatible read for the old MILVUS_URI setting.

    pymilvus also reads MILVUS_URI during import and expects a remote HTTP(S)
    address there. This project uses local Milvus Lite file paths, so remove the
    legacy variable after reading it and keep the project-specific value in
    config only.
    """
    legacy_uri = os.environ.pop("MILVUS_URI", "")
    return os.getenv("RAG_MILVUS_URI", legacy_uri)


_RAW_MILVUS_URI = _consume_legacy_milvus_uri()

# ── 项目根目录 ────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent

# OpenAI 兼容接口地址
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


# ── LLM 配置 ─────────────────────────────────────────────────────
class LLMConfig:
    DEEPSEEK_API_KEY: str = os.getenv("DEEPSEEK_API_KEY", "")
    DASHSCOPE_API_KEY: str = os.getenv("DASHSCOPE_API_KEY", "")
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

    # 模型（DeepSeek 默认值）
    CHAT_MODEL: str = os.getenv("CHAT_MODEL", "deepseek-v4-pro")
    REWRITE_MODEL: str = os.getenv("REWRITE_MODEL", "deepseek-v4-flash")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-v3")

    @classmethod
    def has_deepseek(cls) -> bool:
        return bool(cls.DEEPSEEK_API_KEY and cls.DEEPSEEK_API_KEY.startswith("sk-"))

    @classmethod
    def has_dashscope(cls) -> bool:
        return bool(cls.DASHSCOPE_API_KEY and len(cls.DASHSCOPE_API_KEY) > 10)

    @classmethod
    def has_anthropic(cls) -> bool:
        return bool(cls.ANTHROPIC_API_KEY and cls.ANTHROPIC_API_KEY.startswith("sk-ant"))

    @classmethod
    def has_openai(cls) -> bool:
        return bool(cls.OPENAI_API_KEY and cls.OPENAI_API_KEY.startswith("sk-"))

    @classmethod
    def provider(cls) -> str:
        """返回当前生效的 LLM 提供商"""
        if cls.has_deepseek():
            return "deepseek"
        if cls.has_dashscope():
            return "dashscope"
        if cls.has_anthropic():
            return "anthropic"
        if cls.has_openai():
            return "openai"
        return "none"

    @classmethod
    def validate(cls):
        if cls.provider() == "none":
            raise EnvironmentError(
                "未找到有效的 API Key。\n"
                "请在 .env 文件中设置以下任一项：\n"
                "  DEEPSEEK_API_KEY=sk-xxx  (DeepSeek，推荐)\n"
                "  DASHSCOPE_API_KEY=sk-xxx  (千问，可用于 Embedding)\n"
                "  ANTHROPIC_API_KEY=sk-ant-xxx  (Claude)\n"
                "  OPENAI_API_KEY=sk-xxx  (GPT)\n"
                "参考 .env.example 文件。"
            )


# ── RAG 参数 ─────────────────────────────────────────────────────
class RAGConfig:
    # 文档分块（v2，token 驱动）
    TOKENIZER_NAME: str = os.getenv("TOKENIZER_NAME", "cl100k_base")
    PARENT_TARGET_TOKENS: int = int(os.getenv("PARENT_TARGET_TOKENS", "900"))
    PARENT_MAX_TOKENS: int = int(os.getenv("PARENT_MAX_TOKENS", "1200"))
    PARENT_OVERLAP_TOKENS: int = int(os.getenv("PARENT_OVERLAP_TOKENS", "100"))
    CHILD_TARGET_TOKENS: int = int(os.getenv("CHILD_TARGET_TOKENS", "280"))
    CHILD_MAX_TOKENS: int = int(os.getenv("CHILD_MAX_TOKENS", "360"))
    CHILD_OVERLAP_TOKENS: int = int(os.getenv("CHILD_OVERLAP_TOKENS", "60"))
    ACTIVE_INDEX_VERSION: str = os.getenv("ACTIVE_INDEX_VERSION", "v2")
    MAX_HYDRATED_PARENTS: int = int(os.getenv("MAX_HYDRATED_PARENTS", "6"))

    # 兼容旧字段
    CHUNK_SIZE: int = CHILD_TARGET_TOKENS
    CHUNK_OVERLAP: int = CHILD_OVERLAP_TOKENS

    # 检索参数
    SEMANTIC_TOP_K: int = int(os.getenv("SEMANTIC_TOP_K", "6"))
    BM25_TOP_K: int = int(os.getenv("BM25_TOP_K", "6"))
    FINAL_TOP_K: int = int(os.getenv("FINAL_TOP_K", "4"))
    SEMANTIC_WEIGHT: float = float(os.getenv("SEMANTIC_WEIGHT", "0.7"))
    SIMILARITY_THRESHOLD: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.3"))

    # 对话记忆
    MAX_HISTORY_MESSAGES: int = int(os.getenv("MAX_HISTORY_MESSAGES", "10"))

    # ── 日期感知检索 ──
    DATE_EXTRACTION_ENABLED: bool = os.getenv("DATE_EXTRACTION_ENABLED", "true").lower() == "true"
    DATE_EXTRACTION_LLM_FALLBACK: bool = os.getenv("DATE_EXTRACTION_LLM_FALLBACK", "true").lower() == "true"
    DATE_CACHE_PATH: str = os.getenv("DATE_CACHE_PATH", "data/date_cache.sqlite")
    HARD_FILTER_K_MULTIPLIER: int = int(os.getenv("HARD_FILTER_K_MULTIPLIER", "2"))
    BM25_FILTER_K_MULTIPLIER: int = int(os.getenv("BM25_FILTER_K_MULTIPLIER", "3"))


class RerankConfig:
    ENABLED: bool = os.getenv("RERANK_ENABLED", "true").lower() == "true"
    MODEL: str = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-base")
    TOP_N: int = int(os.getenv("RERANK_TOP_N", "4"))
    BATCH_SIZE: int = int(os.getenv("RERANK_BATCH_SIZE", "16"))
    DEVICE: str = os.getenv("RERANK_DEVICE", "")


# ── 路径配置 ─────────────────────────────────────────────────────
class PathConfig:
    DOCUMENTS_DIR: Path = BASE_DIR / os.getenv("DOCUMENTS_DIR", "data/documents")
    VECTORSTORE_DIR: Path = BASE_DIR / os.getenv("VECTORSTORE_DIR", "data/vectorstore")
    DOCSTORE_DIR: Path = BASE_DIR / os.getenv("DOCSTORE_DIR", "data/docstore")

    @classmethod
    def ensure_dirs(cls):
        cls.DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
        cls.VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
        cls.DOCSTORE_DIR.mkdir(parents=True, exist_ok=True)


def _resolve_milvus_uri() -> str:
    raw_uri = _RAW_MILVUS_URI
    if not raw_uri:
        return str(PathConfig.VECTORSTORE_DIR / "milvus.db")

    if "://" in raw_uri:
        return raw_uri

    path = Path(raw_uri).expanduser()
    if not path.is_absolute():
        path = BASE_DIR / path
    return str(path)


# ── Milvus Lite 配置 ─────────────────────────────────────────────
class MilvusConfig:
    COLLECTION_NAME: str = f"rag_knowledge_base_{RAGConfig.ACTIVE_INDEX_VERSION}_children"
    URI: str = _resolve_milvus_uri()
    CONSISTENCY_LEVEL: str = os.getenv("MILVUS_CONSISTENCY_LEVEL", "Strong")
    PRIMARY_FIELD: str = os.getenv("MILVUS_PRIMARY_FIELD", "pk")
    TEXT_FIELD: str = os.getenv("MILVUS_TEXT_FIELD", "text")
    VECTOR_FIELD: str = os.getenv("MILVUS_VECTOR_FIELD", "vector")
    METADATA_FIELDS: tuple[str, ...] = (
        "doc_id",
        "source",
        "file_path",
        "page",
        "total_pages",
        "doc_version",
        "chunk_role",
        "parent_id",
        "child_id",
        "parent_index",
        "child_index",
        "chunk_index",
        "section_index",
        "section_path",
        "heading_level",
        "is_atomic",
        "token_count",
        "page_range",
        "page_start",
        "page_end",
        "upload_date",
        "doc_date_min",
        "doc_date_max",
        "has_doc_date",
    )
    INDEX_PARAMS: dict = {
        "metric_type": "COSINE",
        "index_type": "FLAT",
        "params": {},
    }
    SEARCH_PARAMS: dict = {
        "metric_type": "COSINE",
        "params": {},
    }


class DocStoreConfig:
    DB_PATH: Path = PathConfig.DOCSTORE_DIR / f"parents_{RAGConfig.ACTIVE_INDEX_VERSION}.sqlite"


# ── 统一导出 ─────────────────────────────────────────────────────
llm_config = LLMConfig()
rag_config = RAGConfig()
rerank_config = RerankConfig()
path_config = PathConfig()
milvus_config = MilvusConfig()
docstore_config = DocStoreConfig()

# 确保目录存在
PathConfig.ensure_dirs()
