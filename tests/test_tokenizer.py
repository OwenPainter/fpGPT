"""Tokenizer contract tests.

The FPGA mirrors the byte <-> token-ID mapping in
fpga/generation_controller.v, so these tests pin the exact formula and the
round-trip behavior the hardware must reproduce.
"""
import json
import tempfile
import unittest
from pathlib import Path

from model.tokenizer import (CharTokenizer, PRINTABLE_START, SPECIAL_TOKENS,
                             PAD_ID, SOS_ID, EOS_ID)
from model.micro_gpt import CharTokenizer as ReexportedTokenizer

ROOT = Path(__file__).resolve().parents[1]


class TokenizerTests(unittest.TestCase):
    def test_reexport_is_same_class(self):
        self.assertIs(ReexportedTokenizer, CharTokenizer)

    def test_special_token_ids(self):
        tok = CharTokenizer(64)
        self.assertEqual((tok.pad_id, tok.sos_id, tok.eos_id), (0, 1, 2))
        self.assertEqual(tok.vocab_size, 64)
        self.assertEqual(len(tok.chars), 61)  # 64 - 3 specials
        # The default vocabulary covers the full printable ASCII range.
        default = CharTokenizer()
        self.assertEqual(default.vocab_size, 98)
        self.assertEqual(len(default.chars), 95)
        self.assertEqual(default.chars[0], ' ')
        self.assertEqual(default.chars[-1], '~')

    def test_encode_decode_roundtrip(self):
        tok = CharTokenizer()
        text = "Hello, world! 123"
        ids = tok.encode(text)
        self.assertEqual(tok.decode(ids), text)
        self.assertTrue(all(3 <= i < tok.vocab_size for i in ids))

    def test_unknown_char_becomes_pad_and_is_skipped(self):
        tok = CharTokenizer()
        ids = tok.encode("a\tb")         # tab (ASCII 9) is outside the vocabulary
        self.assertEqual(ids, [tok.char_to_id['a'], PAD_ID, tok.char_to_id['b']])
        self.assertEqual(tok.decode(ids), "ab")
        self.assertEqual(tok.decode([SOS_ID, tok.char_to_id['a'], EOS_ID],
                                    skip_special=False), "<sos>a<eos>")

    def test_hardware_byte_mapping_formula(self):
        tok = CharTokenizer(64)
        # Exactly the generation_controller.v conditions.
        for b in range(256):
            expected = b - 32 + 3 if 32 <= b < 32 + 64 - 3 else 0
            self.assertEqual(tok.byte_to_id(b), expected, f"byte {b}")
        for i in range(64):
            expected = i - 3 + 32 if i >= 3 else ord('.')
            self.assertEqual(tok.id_to_byte(i), expected, f"id {i}")

    def test_char_mapping_matches_hardware_byte_mapping(self):
        for tok in (CharTokenizer(64), CharTokenizer()):
            for c in tok.chars:
                token_id = tok.char_to_id[c]
                self.assertEqual(tok.byte_to_id(ord(c)), token_id)
                self.assertEqual(tok.id_to_byte(token_id), ord(c))
            self.assertEqual(tok.chars[0], ' ')
            self.assertEqual(ord(tok.chars[-1]), PRINTABLE_START + len(tok.chars) - 1)

    def test_save_load_roundtrip(self):
        tok = CharTokenizer()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tokenizer.json"
            tok.save(path)
            restored = CharTokenizer.load(path)
            self.assertEqual(restored.state_dict(), tok.state_dict())
            self.assertEqual(restored.encode("abc"), tok.encode("abc"))
            self.assertEqual(restored.decode(restored.encode("abc")), "abc")

    def test_rejects_vocab_smaller_than_specials(self):
        with self.assertRaises(ValueError):
            CharTokenizer(vocab_size=2)

    def test_custom_vocab_size_boundaries(self):
        tok = CharTokenizer(16)
        self.assertEqual(len(tok.chars), 13)
        # Last valid byte maps to the last ID; the next byte falls back to pad.
        last_byte = 32 + 13 - 1
        self.assertEqual(tok.byte_to_id(last_byte), 15)
        self.assertEqual(tok.byte_to_id(last_byte + 1), 0)


if __name__ == "__main__":
    unittest.main()
