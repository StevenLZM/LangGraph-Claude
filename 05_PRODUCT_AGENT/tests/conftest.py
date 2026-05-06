from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

TEST_MEMORY_DB = Path("/private/tmp/05_product_agent_pytest_memory.db")
if TEST_MEMORY_DB.exists():
    TEST_MEMORY_DB.unlink()
os.environ.setdefault("MEMORY_DB", str(TEST_MEMORY_DB))
