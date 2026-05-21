from pathlib import Path
import tomllib


def test_streamlit_file_watcher_disabled_for_transformers_optional_imports():
    config_path = Path(__file__).parent.parent / ".streamlit" / "config.toml"
    config = tomllib.loads(config_path.read_text())

    assert config["server"]["fileWatcherType"] == "none"
