#!/usr/bin/env python3
"""Convenience launcher so the package works without installation.

    python host_gui/run.py --mock-style ngram
    python host_gui/run.py --transport serial --serial-port /dev/ttyUSB0
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fgpt_gui.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())