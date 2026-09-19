"""Shared pytest configuration for the fpGPT test suite.

Keeps the repository root and the ``tests/`` directory on ``sys.path`` so the
suite behaves identically under ``pytest``, ``python -m unittest`` and direct
invocation. Also surfaces which external RTL tools are available.
"""

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / "tests"

for path in (str(ROOT), str(TESTS)):
    if path not in sys.path:
        sys.path.insert(0, path)

IVERILOG = shutil.which("iverilog")
VVP = shutil.which("vvp")
VERILATOR = shutil.which("verilator")


def pytest_report_header(config):
    tools = [
        f"iverilog={'yes' if IVERILOG else 'no'}",
        f"vvp={'yes' if VVP else 'no'}",
        f"verilator={'yes' if VERILATOR else 'no'}",
    ]
    return "fpGPT RTL tools: " + ", ".join(tools)
