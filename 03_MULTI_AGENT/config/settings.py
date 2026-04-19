"""进程配置 —— 读取 .env。字段名对齐 03_MULTI_AGENT/.env.example。"""
from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # LLM (DashScope)
    dashscope_api_key: str = ""
    dashscope_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    qwen_max_model: str = "qwen-max"
    qwen_light_model: str = "qwen-plus"
    qwen_embedding_model: str = "text-embedding-v3"

    # 工具
    tavily_api_key: str = ""
    github_token: str = ""
    brave_api_key: str = ""

    # LangSmith
    langchain_tracing_v2: bool = False
    langchain_endpoint: str = "https://api.smith.langchain.com"
    langchain_api_key: str = ""
    langchain_project: str = "insightloop-multi-agent"

    # 路径
    documents_dir: str = str(ROOT / "data" / "documents")
    vectorstore_dir: str = str(ROOT / "data" / "vectorstore")
    reports_dir: str = str(ROOT / "data" / "reports")
    archive_db: str = str(ROOT / "data" / "archive.db")
    checkpointer_db: str = str(ROOT / "data" / "checkpoints.db")

    # 研究流程
    max_reflection_iterations: int = 3
    default_research_depth: str = "standard"
    max_parallel_researchers: int = 4
    research_timeout_seconds: int = 180

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8080
    sse_retry_ms: int = 3000

    # Dogfooding
    use_internal_mcp_for_kb: bool = False

    # MCP 配置
    mcp_config_path: str = str(ROOT / ".mcp.json")

    # 代理（macOS 系统代理在 127.0.0.1:7890 时填这里；为空=不走代理）
    # 仅作用于：外部 MCP 子进程、httpx 客户端（DashScope/Tavily/ArXiv/GitHub 走主机直连，
    # 国内 LLM/搜索一般无需代理；Brave 等海外服务才需要）
    https_proxy: str = ""
    http_proxy: str = ""


settings = Settings()
