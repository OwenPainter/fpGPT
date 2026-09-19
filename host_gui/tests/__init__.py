"""Tests run with the standard library only::

    python -m unittest discover -s host_gui/tests -t host_gui -v

(``-t host_gui`` puts ``fgpt_gui`` on the import path.)
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make ``import fgpt_gui`` work regardless of how discovery is invoked.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))