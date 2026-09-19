"""Parse the compiler-generated ``board_params.vh`` contract.

``compile.py`` emits ``build/rtl/board_params.vh``; ``fpga/build.tcl`` copies it
over ``fpga/board_params.vh``. Rather than duplicate the model geometry in the
GUI, we read the same ``FPGPT_*`` macros the RTL uses:

    `define FPGPT_D_MODEL      64
    `define FPGPT_ROM_MEM_FILE "weights_unified.hex"

Unknown macros are ignored so new RTL parameters do not break the GUI.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

# `define FPGPT_NAME value   (a trailing // comment is stripped first)
_DEFINE_RE = re.compile(r"`define\s+FPGPT_([A-Za-z0-9_]+)\s+(.+?)\s*$")

# Macro suffix -> dataclass field for plain integer/string parameters.
_FIELD_MAP = {
    "DATA_WIDTH": "data_width",
    "D_MODEL": "d_model",
    "NUM_HEADS": "num_heads",
    "MAX_SEQ_LEN": "max_seq_len",
    "NUM_LAYERS": "num_layers",
    "D_FF": "d_ff",
    "VOCAB_SIZE": "vocab_size",
    "W_ADDR_WIDTH": "w_addr_width",
    "ROM_DEPTH": "rom_depth",
    "ROM_MEM_FILE": "rom_mem_file",
    "GEN_TOKENS": "gen_tokens",
    "SYS_CLK_HZ": "sys_clk_hz",
    "BAUD_RATE": "baud_rate",
}

# Macro suffix -> key in the ``shifts`` mapping.
_SHIFT_MAP = {
    "Q_SHIFT": "q_shift",
    "K_SHIFT": "k_shift",
    "V_SHIFT": "v_shift",
    "OUT_SHIFT": "out_shift",
    "SCORE_MULT": "score_mult",
    "SCORE_SHIFT": "score_shift",
    "LN_SHIFT": "ln_shift",
    "FC1_SHIFT": "fc1_shift",
    "FC2_SHIFT": "fc2_shift",
    "LM_SHIFT": "lm_shift",
}

# Values mirror the checked-in placeholder in fpga/board_params.vh so the GUI
# still starts (with a warning) if the file is missing.
_DEFAULTS = {
    "data_width": 8,
    "d_model": 64,
    "num_heads": 4,
    "max_seq_len": 64,
    "num_layers": 4,
    "d_ff": 256,
    "vocab_size": 64,
    "w_addr_width": 18,
    "rom_depth": 212352,
    "rom_mem_file": "weights_unified.hex",
    "gen_tokens": 16,
    "sys_clk_hz": 150000000,
    "baud_rate": 115200,
}

_DEFAULT_SHIFTS = {
    "q_shift": 6,
    "k_shift": 6,
    "v_shift": 7,
    "out_shift": 7,
    "score_mult": 262144,
    "score_shift": 28,
    "ln_shift": 7,
    "fc1_shift": 6,
    "fc2_shift": 8,
    "lm_shift": 6,
}


@dataclass
class BoardParams:
    """The subset of board parameters the host cares about."""

    data_width: int = 8
    d_model: int = 64
    num_heads: int = 4
    max_seq_len: int = 64
    num_layers: int = 4
    d_ff: int = 256
    vocab_size: int = 64
    w_addr_width: int = 18
    rom_depth: int = 212352
    rom_mem_file: str = "weights_unified.hex"
    gen_tokens: int = 16
    sys_clk_hz: int = 150000000
    baud_rate: int = 115200
    shifts: dict = field(default_factory=dict)

    # Where the values came from (a path, or None when defaults were used).
    source: str | None = None

    @property
    def char_min(self) -> int:
        """Lowest supported printable character (mirrors CharTokenizer)."""
        return 32

    @property
    def char_max(self) -> int:
        """Highest supported printable character (inclusive).

        Mirrors ``generation_controller.byte_to_id``:
        IDs 3..V-1 map to ASCII 32..32+(V-3)-1.
        """
        return 32 + self.vocab_size - 3 - 1

    def to_dict(self) -> dict:
        data = asdict(self)
        data["char_min"] = self.char_min
        data["char_max"] = self.char_max
        return data


def _parse_value(raw: str):
    """Turn a macro body into an int, a string, or None if unparseable."""
    raw = raw.strip()
    if raw.startswith('"') and raw.endswith('"') and len(raw) >= 2:
        return raw[1:-1]
    token = raw.split()[0] if raw.split() else ""
    token = token.strip()
    if not token:
        return None
    try:
        return int(token.replace("_", ""), 0)
    except ValueError:
        return None


def parse_board_params(path) -> BoardParams:
    """Parse a ``board_params.vh`` file into :class:`BoardParams`.

    Unreadable/missing files raise ``FileNotFoundError``; malformed values are
    skipped and fall back to defaults (the file is generated, not trusted).
    """
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    params = BoardParams(shifts=dict(_DEFAULT_SHIFTS), source=str(path))

    for line in text.splitlines():
        line = line.split("//", 1)[0]
        match = _DEFINE_RE.search(line)
        if not match:
            continue
        name, raw = match.group(1), match.group(2)
        value = _parse_value(raw)
        if value is None:
            continue
        if name in _FIELD_MAP:
            setattr(params, _FIELD_MAP[name], value)
        elif name in _SHIFT_MAP:
            params.shifts[_SHIFT_MAP[name]] = value

    return params


def _repo_root() -> Path:
    # host_gui/fgpt_gui/board_params.py -> repo root is two levels up.
    return Path(__file__).resolve().parents[2]


def find_board_params(start=None) -> Path | None:
    """Search the usual locations for a generated board_params.vh."""
    candidates = []
    if start is not None:
        candidates.append(Path(start))
    root = _repo_root()
    candidates += [
        Path.cwd() / "fpga" / "board_params.vh",
        Path.cwd() / "build" / "rtl" / "board_params.vh",
        root / "fpga" / "board_params.vh",
        root / "build" / "rtl" / "board_params.vh",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def load_board_params(path=None) -> tuple[BoardParams, list[str]]:
    """Load board parameters, returning ``(params, warnings)``.

    ``path`` may be an explicit file; otherwise the standard locations are
    searched. If nothing is found, defaults are returned with a warning.
    """
    warnings: list[str] = []
    target = Path(path) if path else find_board_params()
    if target is None:
        warnings.append(
            "board_params.vh not found; using built-in defaults. "
            "Run compile.py or pass --board-params."
        )
        return BoardParams(shifts=dict(_DEFAULT_SHIFTS)), warnings
    try:
        return parse_board_params(target), warnings
    except OSError as exc:
        warnings.append(f"could not read {target}: {exc}; using defaults")
        return BoardParams(shifts=dict(_DEFAULT_SHIFTS)), warnings