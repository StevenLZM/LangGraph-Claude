from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).parent.parent))


def test_build_rag_invoke_config_includes_session_metadata():
    import app

    config = app.build_rag_invoke_config("session-123")

    assert config["configurable"]["session_id"] == "session-123"
    assert config["metadata"]["session_id"] == "session-123"
    assert config["metadata"]["application"] == "01_RAG"
    assert config["metadata"]["interface"] == "streamlit"
    assert config["tags"] == ["01-rag", "streamlit"]
