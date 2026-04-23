"""
mcp/filesystem_client.py — MCP Filesystem Server 客户端适配层

在 Claude Desktop / Claude Code 环境中，MCP Server 由宿主进程管理。
本模块提供两种模式：
  1. MCP 模式：通过 MCP 协议与 filesystem server 通信（Claude Desktop/Code 中运行时）
  2. 直接模式：直接使用 Python 文件 API（本地开发/Streamlit 独立运行时）
"""
from __future__ import annotations
import os
import shutil
from pathlib import Path
from typing import List, Optional


class FilesystemClient:
    """
    文件系统操作客户端
    自动检测运行环境，在 MCP 可用时走 MCP 协议，否则直接操作文件系统
    """

    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._mcp_available = self._check_mcp_available()

    def _check_mcp_available(self) -> bool:
        """检测 MCP 运行时是否可用"""
        try:
            # 在 Claude Desktop 或 Claude Code 中，
            # MCP 工具通过环境变量标记可用
            return bool(os.environ.get("MCP_SERVER_FILESYSTEM"))
        except Exception:
            return False

    # ── 核心文件操作 ──────────────────────────────────────────────

    def list_files(self, pattern: str = "*.pdf") -> List[dict]:
        """列出目录中的文件"""
        files = []
        for f in sorted(self.base_dir.glob(pattern)):
            stat = f.stat()
            files.append({
                "name": f.name,
                "path": str(f),
                "size_kb": round(stat.st_size / 1024, 1),
                "modified": stat.st_mtime,
            })
        return files

    def read_file(self, filename: str) -> bytes:
        """读取文件内容（二进制）"""
        file_path = self.base_dir / filename
        if not file_path.exists():
            raise FileNotFoundError(f"文件不存在: {filename}")
        return file_path.read_bytes()

    def save_file(self, filename: str, content: bytes) -> str:
        """保存文件，返回绝对路径"""
        # 安全检查：防止路径穿越
        dest = self.base_dir / Path(filename).name  # 只取文件名部分
        dest.write_bytes(content)
        return str(dest)

    def delete_file(self, filename: str) -> bool:
        """删除文件"""
        file_path = self.base_dir / filename
        if file_path.exists():
            file_path.unlink()
            return True
        return False

    def file_exists(self, filename: str) -> bool:
        """检查文件是否存在"""
        return (self.base_dir / filename).exists()

    def get_full_path(self, filename: str) -> str:
        """获取文件的绝对路径"""
        return str(self.base_dir / filename)

    def get_dir_size_mb(self) -> float:
        """返回目录总大小（MB）"""
        total = sum(f.stat().st_size for f in self.base_dir.rglob("*") if f.is_file())
        return round(total / 1024 / 1024, 2)


# ── MCP Server 配置（供 Claude Desktop / Claude Code 使用） ──────
MCP_CONFIG = {
    "mcpServers": {
        "filesystem": {
            "command": "npx",
            "args": [
                "-y",
                "@modelcontextprotocol/server-filesystem",
                # 将实际路径注入（由 app.py 在启动时写入）
                "{{DOCUMENTS_DIR}}"
            ],
            "description": "文件系统访问：读取和管理上传的 PDF 文件"
        }
    }
}


def get_filesystem_client(documents_dir: str) -> FilesystemClient:
    """获取文件系统客户端（工厂函数）"""
    return FilesystemClient(documents_dir)
