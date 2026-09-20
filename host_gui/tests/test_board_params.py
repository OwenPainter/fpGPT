from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from fgpt_gui.board_params import (
    BoardParams,
    find_board_params,
    load_board_params,
    parse_board_params,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CHECKED_IN = REPO_ROOT / "fpga" / "board_params.vh"


class BoardParamsParserTests(unittest.TestCase):
    def test_parses_checked_in_contract(self):
        if not CHECKED_IN.is_file():
            self.skipTest("fpga/board_params.vh not present")
        params = parse_board_params(CHECKED_IN)
        self.assertEqual(params.data_width, 8)
        self.assertEqual(params.d_model, 64)
        self.assertEqual(params.vocab_size, 64)
        self.assertEqual(params.max_seq_len, 64)
        self.assertEqual(params.gen_tokens, 16)
        self.assertEqual(params.baud_rate, 115200)
        self.assertEqual(params.rom_mem_file, "weights_unified.hex")
        # This header is regenerated from the selected checkpoint; shifts vary.
        self.assertIsInstance(params.shifts["q_shift"], int)
        self.assertEqual(params.source, str(CHECKED_IN))

    def test_char_range_matches_tokenizer(self):
        # vocab=64 -> IDs 3..63 -> ASCII 32..92.
        params = BoardParams(vocab_size=64)
        self.assertEqual(params.char_min, 32)
        self.assertEqual(params.char_max, 92)
        # vocab=98 -> full printable range, as CharTokenizer defaults.
        params = BoardParams(vocab_size=98)
        self.assertEqual(params.char_max, 126)

    def test_strings_comments_and_unknown_macros(self):
        text = "\n".join([
            "// a comment line",
            "`define FPGPT_DATA_WIDTH 8   // inline comment",
            "`define FPGPT_ROM_MEM_FILE \"weights/engine/weights_unified.hex\"",
            "`define FPGPT_ROM_DEPTH 1_024",
            "`define FPGPT_Q_SHIFT 8",
            "`define FPGPT_TOTALLY_NEW 42",
            "`define FPGPT_D_MODEL (2*32)",
        ])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "board_params.vh"
            path.write_text(text)
            params = parse_board_params(path)
        self.assertEqual(params.data_width, 8)
        self.assertEqual(params.rom_mem_file, "weights/engine/weights_unified.hex")
        self.assertEqual(params.rom_depth, 1024)
        self.assertEqual(params.shifts["q_shift"], 8)
        # Unknown macros are ignored, and unparseable expressions are skipped
        # (default kept) rather than crashing.
        self.assertEqual(params.d_model, 64)

    def test_missing_file_is_warned_but_not_fatal(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "nope.vh"
            params, warnings = load_board_params(missing)
        self.assertTrue(warnings)
        self.assertIsInstance(params, BoardParams)
        self.assertEqual(params.gen_tokens, 16)

    def test_find_board_params_locates_repo_file(self):
        found = find_board_params()
        if CHECKED_IN.is_file():
            self.assertIsNotNone(found)


if __name__ == "__main__":
    unittest.main()
