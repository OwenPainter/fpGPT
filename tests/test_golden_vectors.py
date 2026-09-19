"""Golden-vector regression for the compiler's numeric ABI.

Builds a small deterministic Micro-GPT, runs the real compile pipeline
(quantize -> fixed contract -> memory layout -> weight export -> board
parameters), and freezes the resulting activation formats, requantization
shifts, byte addresses and ROM lengths. Any change to the numeric contract
shows up as a diff here even when the RTL sims still pass.

Regenerate after an intentional contract change:

    FPGPT_UPDATE_GOLDEN=1 python -m pytest tests/test_golden_vectors.py

Run: python -m pytest tests/test_golden_vectors.py -v
"""

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="golden vectors require torch")

from compiler.fixed_export import export_fixed  # noqa: E402
from compiler.mif_writer import export_all_weights  # noqa: E402
from compiler.quantizer import quantize_model  # noqa: E402
from model.micro_gpt import MicroGPT  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "tests" / "golden" / "tiny_model.json"

# Small, explicit config so the golden vectors do not depend on DEFAULT_CONFIG
# (which tracks the trained production model and is expected to change).
TINY_CONFIG = {
    "vocab_size": 16,
    "max_seq_len": 4,
    "d_model": 8,
    "num_heads": 2,
    "num_layers": 1,
    "d_ff": 16,
    "dropout": 0.0,
}
SEED = 20240919


def build_tiny_model():
    """Construct the tiny model with weights fixed by a stable NumPy RNG.

    Torch's RNG stream can differ between releases, so the golden vectors must
    not depend on it. Filling every parameter from ``default_rng(SEED)`` keeps
    the test reproducible across torch/numpy versions and platforms.
    """
    model = MicroGPT(dict(TINY_CONFIG)).eval()
    rng = np.random.default_rng(SEED)
    with torch.no_grad():
        for name, param in model.named_parameters():
            shape = tuple(param.shape)
            if "ln" in name and name.endswith("weight"):
                values = 1.0 + rng.uniform(-0.1, 0.1, size=shape)
            elif name.endswith("bias"):
                values = rng.uniform(-0.02, 0.02, size=shape)
            else:
                values = rng.uniform(-0.05, 0.05, size=shape)
            param.copy_(torch.as_tensor(values, dtype=param.dtype))
    return model


def _hex_token_count(path: Path) -> int:
    return len(path.read_text().split())


def collect_vectors(workdir: Path) -> dict:
    model = build_tiny_model()

    ir = quantize_model(model, TINY_CONFIG, bit_width=8)
    ir_engine = quantize_model(model, TINY_CONFIG, bit_width=8,
                               engine_compatible=True)

    weights_dir = workdir / "weights"
    export_all_weights(ir, weights_dir, fmt="hex")
    export_all_weights(ir_engine, weights_dir / "engine", fmt="hex",
                       unified_only=True)

    manifest = export_fixed(ir, workdir)

    mixed_hex = weights_dir / "weights_unified.hex"
    engine_hex = weights_dir / "engine" / "weights_unified.hex"
    depth = ir_engine.total_weight_bytes
    w_addr_width = max(1, (depth - 1).bit_length())

    return {
        "seed": SEED,
        "config": TINY_CONFIG,
        "formats": ir.formats,
        "total_params": ir.total_params,
        "mixed_width_bytes": ir.total_weight_bytes,
        "engine_bytes": ir_engine.total_weight_bytes,
        "engine_hex_tokens": _hex_token_count(engine_hex),
        "mixed_hex_tokens": _hex_token_count(mixed_hex),
        "rom_depth": depth,
        "w_addr_width": w_addr_width,
        "manifest": manifest,
    }


def _diff(expected: dict, current: dict, prefix: str = "") -> list:
    """Return a flat list of human-readable differences between two trees."""
    problems = []
    keys = sorted(set(expected) | set(current))
    for key in keys:
        path = f"{prefix}.{key}" if prefix else str(key)
        if key not in expected:
            problems.append(f"{path}: unexpected (current={current[key]!r})")
        elif key not in current:
            problems.append(f"{path}: missing (expected={expected[key]!r})")
        else:
            a, b = expected[key], current[key]
            if isinstance(a, dict) and isinstance(b, dict):
                problems.extend(_diff(a, b, path))
            elif a != b:
                problems.append(f"{path}: expected {a!r}, got {b!r}")
    return problems


def test_golden_vectors_are_frozen():
    with tempfile.TemporaryDirectory(prefix="fpgpt-golden-") as tmp:
        current = collect_vectors(Path(tmp))

    if os.environ.get("FPGPT_UPDATE_GOLDEN") == "1":
        GOLDEN.parent.mkdir(parents=True, exist_ok=True)
        GOLDEN.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n")
        pytest.skip(f"regenerated {GOLDEN.relative_to(ROOT)}")

    assert GOLDEN.exists(), (
        f"{GOLDEN.relative_to(ROOT)} is missing; regenerate with "
        "FPGPT_UPDATE_GOLDEN=1"
    )
    expected = json.loads(GOLDEN.read_text())

    problems = _diff(expected, current)
    assert not problems, "compiler golden vectors changed:\n  " + "\n  ".join(problems)


def test_engine_image_is_uniform_width():
    """The engine ROM must be exactly one byte per parameter (the contract the
    board ROM relies on)."""
    with tempfile.TemporaryDirectory(prefix="fpgpt-golden-") as tmp:
        current = collect_vectors(Path(tmp))
    assert current["engine_bytes"] == current["engine_hex_tokens"]
    assert current["mixed_width_bytes"] == current["mixed_hex_tokens"]
    assert current["rom_depth"] == current["engine_bytes"]
