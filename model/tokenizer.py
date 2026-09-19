"""
fpGPT Model — Tokenizer (single source of truth for text <-> token IDs)

The mapping is deliberately hardware-friendly so the FPGA can reproduce it in
combinational logic:

    ID 0            <pad>   (unmapped / padding)
    ID 1            <sos>   (start of sequence)
    ID 2            <eos>   (end of sequence)
    ID 3 .. V-1     printable ASCII ' ' (32) upward, one ID per character

For a vocabulary of size V there are V-3 characters, covering ASCII
32 .. 32+(V-3)-1. The board mirror lives in fpga/generation_controller.v
(byte_to_id / id_to_byte); tests/test_tokenizer.py locks the two together.

Tokenizers are serializable so a checkpoint records exactly the mapping used
during training, and inference/compilation can reload it.
"""

from __future__ import annotations

import json
from pathlib import Path

SPECIAL_TOKENS = {"<pad>": 0, "<sos>": 1, "<eos>": 2}
NUM_SPECIAL = len(SPECIAL_TOKENS)
PRINTABLE_START = 32    # ' '
PRINTABLE_END = 127     # exclusive, so '~' (126) is the last candidate
PAD_ID = SPECIAL_TOKENS["<pad>"]
SOS_ID = SPECIAL_TOKENS["<sos>"]
EOS_ID = SPECIAL_TOKENS["<eos>"]


class CharTokenizer:
    """Character-level tokenizer with an explicit, serializable vocabulary.

    Args:
        vocab_size: Total number of token IDs (special tokens included).
        special_tokens: Mapping of special-token name to ID.
        chars: Explicit character list. Defaults to the printable ASCII range
            truncated to ``vocab_size - len(special_tokens)`` entries.
    """

    def __init__(self, vocab_size: int = 98, special_tokens: dict = None,
                 chars: list = None):
        self.vocab_size = int(vocab_size)
        self.special_tokens = dict(special_tokens or SPECIAL_TOKENS)
        num_special = len(self.special_tokens)

        if chars is None:
            capacity = self.vocab_size - num_special
            if capacity < 0:
                raise ValueError("vocab_size smaller than the special tokens")
            printable = [chr(i) for i in range(PRINTABLE_START, PRINTABLE_END)]
            chars = printable[:capacity]
        self.chars = list(chars)

        if num_special + len(self.chars) > self.vocab_size:
            raise ValueError("vocabulary does not fit in vocab_size")

        self.char_to_id = {c: i + num_special for i, c in enumerate(self.chars)}
        self.id_to_char = {i + num_special: c for i, c in enumerate(self.chars)}
        for tok, idx in self.special_tokens.items():
            self.id_to_char[idx] = tok

    # ── Vocabulary properties ──
    @property
    def pad_id(self) -> int:
        return self.special_tokens["<pad>"]

    @property
    def sos_id(self) -> int:
        return self.special_tokens["<sos>"]

    @property
    def eos_id(self) -> int:
        return self.special_tokens["<eos>"]

    @property
    def num_special(self) -> int:
        return len(self.special_tokens)

    def __len__(self) -> int:
        return self.vocab_size

    # ── Text <-> IDs ──
    def encode(self, text: str) -> list:
        """Map a string to token IDs; unknown characters become <pad>."""
        return [self.char_to_id.get(c, self.pad_id) for c in text]

    def decode(self, ids, skip_special: bool = True) -> str:
        """Map token IDs back to a string.

        Args:
            ids: Iterable of token IDs.
            skip_special: Drop special tokens (default) or render their names.
        """
        out = []
        for i in ids:
            i = int(i)
            c = self.id_to_char.get(i)
            if c is None:
                continue
            if c in self.special_tokens:
                if not skip_special:
                    out.append(c)
                continue
            out.append(c)
        return "".join(out)

    # ── Hardware mirror (fpga/generation_controller.v) ──
    def byte_to_id(self, byte: int) -> int:
        """Match the board's byte_to_id function exactly."""
        if PRINTABLE_START <= byte < PRINTABLE_START + self.vocab_size - self.num_special:
            return byte - PRINTABLE_START + self.num_special
        return self.pad_id

    def id_to_byte(self, token_id: int) -> int:
        """Match the board's id_to_byte function exactly ('.' for specials)."""
        if token_id >= self.num_special:
            return token_id - self.num_special + PRINTABLE_START
        return ord('.')

    # ── Serialization ──
    def state_dict(self) -> dict:
        return {
            "type": "char",
            "vocab_size": self.vocab_size,
            "special_tokens": self.special_tokens,
            "chars": self.chars,
        }

    @classmethod
    def from_state_dict(cls, state: dict) -> "CharTokenizer":
        if state.get("type", "char") != "char":
            raise ValueError(f"unsupported tokenizer type: {state.get('type')}")
        return cls(vocab_size=state["vocab_size"],
                   special_tokens=state["special_tokens"],
                   chars=state["chars"])

    def save(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.state_dict(), indent=2))

    @classmethod
    def load(cls, path) -> "CharTokenizer":
        return cls.from_state_dict(json.loads(Path(path).read_text()))
