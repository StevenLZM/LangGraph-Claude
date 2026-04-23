"""
config.py — 全局配置中心
从环境变量加载，提供合理默认值，所有模块均从此处导入配置

支持的 LLM 提供商（优先级从高到低）：
  1. DashScope (千问/Qwen) — DASHSCOPE_API_KEY
  2. Anthropic (Claude)    — ANTHROPIC_API_KEY
  3. OpenAI (GPT)          — OPENAI_API_KEY
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# 加载 .env 文件（优先级低于系统环境变量）
load_dotenv()

# ── 项目根目录 ────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent

# DashScope OpenAI 兼容接口地址
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


# ── LLM 配置 ─────────────────────────────────────────────────────
class LLMConfig:
    DASHSCOPE_API_KEY: str = os.getenv("DASHSCOPE_API_KEY", "")
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

    # 模型（DashScope 默认值）
    CHAT_MODEL: str = os.getenv("CHAT_MODEL", "qwen-plus")
    REWRITE_MODEL: str = os.getenv("REWRITE_MODEL", "qwen-turbo")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-v3")

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
                "  DASHSCOPE_API_KEY=sk-xxx  (千问，推荐)\n"
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


# ── ChromaDB 配置 ─────────────────────────────────────────────────
class ChromaConfig:
    COLLECTION_NAME: str = f"rag_knowledge_base_{RAGConfig.ACTIVE_INDEX_VERSION}_children"
    PERSIST_DIRECTORY: str = str(PathConfig.VECTORSTORE_DIR)


class DocStoreConfig:
    DB_PATH: Path = PathConfig.DOCSTORE_DIR / f"parents_{RAGConfig.ACTIVE_INDEX_VERSION}.sqlite"


# ── 统一导出 ─────────────────────────────────────────────────────
llm_config = LLMConfig()
rag_config = RAGConfig()
path_config = PathConfig()
chroma_config = ChromaConfig()
docstore_config = DocStoreConfig()

# 确保目录存在
PathConfig.ensure_dirs()
