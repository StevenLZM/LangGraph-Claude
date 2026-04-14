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
    # 文档分块
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "500"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "50"))

    # 检索参数
    SEMANTIC_TOP_K: int = int(os.getenv("SEMANTIC_TOP_K", "6"))
    BM25_TOP_K: int = int(os.getenv("BM25_TOP_K", "6"))
    FINAL_TOP_K: int = int(os.getenv("FINAL_TOP_K", "4"))
    SEMANTIC_WEIGHT: float = float(os.getenv("SEMANTIC_WEIGHT", "0.6"))
    SIMILARITY_THRESHOLD: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.3"))

    # 对话记忆
    MAX_HISTORY_MESSAGES: int = int(os.getenv("MAX_HISTORY_MESSAGES", "10"))


# ── 路径配置 ─────────────────────────────────────────────────────
class PathConfig:
    DOCUMENTS_DIR: Path = BASE_DIR / os.getenv("DOCUMENTS_DIR", "data/documents")
    VECTORSTORE_DIR: Path = BASE_DIR / os.getenv("VECTORSTORE_DIR", "data/vectorstore")

    @classmethod
    def ensure_dirs(cls):
        cls.DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
        cls.VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)


# ── ChromaDB 配置 ─────────────────────────────────────────────────
class ChromaConfig:
    COLLECTION_NAME: str = "rag_knowledge_base"
    PERSIST_DIRECTORY: str = str(PathConfig.VECTORSTORE_DIR)


# ── 统一导出 ─────────────────────────────────────────────────────
llm_config = LLMConfig()
rag_config = RAGConfig()
path_config = PathConfig()
chroma_config = ChromaConfig()

# 确保目录存在
PathConfig.ensure_dirs()
