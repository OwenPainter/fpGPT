"""HDL lint gate for ``hdl/`` and ``fpga/``.

Two independent checks:

1. **Icarus elaboration** — every module is elaborated as a top with the whole
   design on the command line. This catches port/instantiation/include errors
   that targeted testbenches never reach.
2. **Verilator lint regression** — ``--lint-only -Wall`` runs over the design
   and is compared against a checked-in baseline, so new width/latch/unused
   warnings fail the build. Latch-class rules are never allowed in the
   baseline.

Regenerate the Verilator baseline after an intentional RTL change:

    FPGPT_UPDATE_LINT_BASELINE=1 python -m pytest tests/test_hdl_lint.py

Run: python -m pytest tests/test_hdl_lint.py -v
"""

import json
import os
import re
import shutil
import subprocess
import tempfile
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
HDL_DIRS = [ROOT / "hdl", ROOT / "fpga"]
STUB = ROOT / "tests" / "hdl_stubs" / "altera_pll.v"
BASELINE = ROOT / "tests" / "golden" / "verilator_baseline.json"

# Rules that must never be baselined: they indicate hardware that cannot work.
HARD_FAIL_RULES = {
    "LATCH",
    "MULTIDRIVEN",
    "COMBDLY",
    "BLKSEQ",
    "IMPLICIT",
    "CASEINCOMPLETE",
    "PINMISSING",
    "PINNOTFOUND",
}

_MODULE_RE = re.compile(r"^\s*module\s+([A-Za-z_]\w*)", re.MULTILINE)
_WARNING_RE = re.compile(r"%Warning-([A-Z]+):\s+([^\s:][^:]*):\d+:")
_VERSION_RE = re.compile(r"Verilator\s+([0-9]+\.[0-9]+)")


def hdl_sources():
    files = []
    for directory in HDL_DIRS:
        files.extend(sorted(directory.glob("*.v")))
    return files


def hdl_modules():
    modules = []
    for path in hdl_sources():
        match = _MODULE_RE.search(path.read_text(encoding="utf-8"))
        if match:
            modules.append((match.group(1), path))
    return modules


def _rel(path):
    try:
        return str(Path(path).relative_to(ROOT))
    except ValueError:
        return str(path)


def _iverilog_available():
    return bool(shutil.which("iverilog") and shutil.which("vvp"))


@pytest.mark.skipif(not _iverilog_available(), reason="requires Icarus Verilog")
def test_iverilog_elaborates_every_module():
    sources = [str(STUB)] + [str(s) for s in hdl_sources()]
    modules = hdl_modules()
    assert modules, "no HDL modules found under hdl/ or fpga/"

    failures = []
    with tempfile.TemporaryDirectory(prefix="fpgpt-elab-") as tmp:
        for name, path in modules:
            result = subprocess.run(
                ["iverilog", "-g2012", "-s", name, "-Ifpga", "-Ihdl",
                 "-o", str(Path(tmp) / name)] + sources,
                capture_output=True, text=True,
            )
            if result.returncode != 0:
                failures.append(
                    f"{_rel(path)} (top={name}):\n{result.stderr.strip()}"
                )

    assert not failures, "modules failed to elaborate:\n\n" + "\n\n".join(failures)


def run_verilator():
    cmd = [
        "verilator", "--lint-only", "-Wall", "--timing", "-Wno-fatal",
        "-Wno-MULTITOP", "-Ifpga", "-Ihdl",
        str(STUB.relative_to(ROOT)),
    ] + [_rel(s) for s in hdl_sources()]
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)


def parse_warnings(text):
    counts = Counter()
    for rule, path in _WARNING_RE.findall(text):
        counts[f"{_rel(path)}::{rule}"] += 1
    return counts


def verilator_version():
    result = subprocess.run(["verilator", "--version"],
                            capture_output=True, text=True)
    match = _VERSION_RE.search(result.stdout + result.stderr)
    return match.group(1) if match else "unknown"


@pytest.mark.skipif(not shutil.which("verilator"), reason="requires Verilator")
def test_verilator_lint_against_baseline():
    result = run_verilator()
    output = result.stdout + result.stderr

    errors = [line for line in output.splitlines() if line.startswith("%Error")]
    assert not errors, "Verilator reported errors:\n" + "\n".join(errors)

    current = parse_warnings(output)
    hard = {key: count for key, count in current.items()
            if key.rsplit("::", 1)[-1] in HARD_FAIL_RULES}
    assert not hard, "Verilator found latch-class issues:\n" + "\n".join(
        f"  {key} x{count}" for key, count in sorted(hard.items())
    )

    version = verilator_version()

    if os.environ.get("FPGPT_UPDATE_LINT_BASELINE") == "1":
        BASELINE.parent.mkdir(parents=True, exist_ok=True)
        BASELINE.write_text(json.dumps(
            {"verilator": version, "warnings": dict(sorted(current.items()))},
            indent=2, sort_keys=True,
        ) + "\n")
        pytest.skip(f"regenerated {_rel(BASELINE)}")

    assert BASELINE.exists(), (
        f"{_rel(BASELINE)} is missing; regenerate with "
        "FPGPT_UPDATE_LINT_BASELINE=1"
    )
    baseline = json.loads(BASELINE.read_text())
    known = baseline.get("warnings", {})

    new_keys = sorted(set(current) - set(known))
    increased = sorted(
        key for key, count in current.items()
        if key in known and count > known[key]
    )

    problems = []
    if new_keys:
        problems.append("new Verilator warnings:\n" + "\n".join(
            f"  {key} x{current[key]}" for key in new_keys
        ))
    if increased:
        if baseline.get("verilator") == version:
            problems.append("Verilator warning counts increased:\n" + "\n".join(
                f"  {key}: {known[key]} -> {current[key]}" for key in increased
            ))
        else:
            print(f"note: Verilator {version} differs from baseline "
                  f"{baseline.get('verilator')}; ignoring count changes")

    assert not problems, "\n".join(problems)
