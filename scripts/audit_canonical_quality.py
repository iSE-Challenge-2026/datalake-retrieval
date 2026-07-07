"""CLI wrapper for canonical artifact quality audits."""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.canonical.audit import *  # noqa: E402,F403
from src.canonical.audit import main  # noqa: E402


if __name__ == "__main__":
    main()
