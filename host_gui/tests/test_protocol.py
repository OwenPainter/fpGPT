from __future__ import annotations

import unittest

from fgpt_gui.protocol import (
    ETX,
    STX,
    Command,
    Frame,
    FrameDecoder,
    FrameError,
    checksum,
    encode_frame,
)


class ProtocolTests(unittest.TestCase):
    def test_round_trip_empty_payload(self):
        raw = encode_frame(Command.PING)
        self.assertEqual(raw[0], STX)
        self.assertEqual(raw[-1], ETX)
        self.assertEqual(len(raw), 5)
        frames = FrameDecoder().feed(raw)
        self.assertEqual(frames, [Frame(int(Command.PING), b"")])

    def test_round_trip_payload(self):
        payload = b"hello world"
        raw = encode_frame(Command.PROMPT, payload)
        frames = FrameDecoder().feed(raw)
        self.assertEqual(len(frames), 1)
        self.assertEqual(frames[0].command, int(Command.PROMPT))
        self.assertEqual(frames[0].payload, payload)
        self.assertEqual(frames[0].cmd, Command.PROMPT)

    def test_checksum_is_xor(self):
        self.assertEqual(checksum(b"\x01\x02\x03"), 0x00)
        self.assertEqual(checksum(b"\x02\x03"), 0x01)

    def test_multiple_frames(self):
        stream = encode_frame(Command.TOKEN, b"a") + encode_frame(Command.TOKEN, b"b")
        frames = FrameDecoder().feed(stream)
        self.assertEqual([f.payload for f in frames], [b"a", b"b"])

    def test_byte_by_byte_streaming(self):
        raw = encode_frame(Command.PROMPT, b"abc")
        decoder = FrameDecoder()
        collected = []
        for byte in raw:
            collected.extend(decoder.feed(bytes([byte])))
        self.assertEqual(len(collected), 1)
        self.assertEqual(collected[0].payload, b"abc")

    def test_garbage_resynchronises(self):
        raw = encode_frame(Command.TOKEN, b"z")
        stream = b"\x00\x11\x22" + raw + b"\xff\xfe"
        decoder = FrameDecoder()
        frames = decoder.feed(stream)
        self.assertEqual([f.payload for f in frames], [b"z"])
        self.assertGreaterEqual(decoder.errors, 1)

    def test_bad_checksum_is_dropped_and_next_frame_parsed(self):
        good = encode_frame(Command.TOKEN, b"g")
        bad = bytearray(encode_frame(Command.TOKEN, b"b"))
        bad[3] ^= 0xFF  # corrupt payload -> checksum mismatch
        frames = FrameDecoder().feed(bytes(bad) + good)
        self.assertEqual([f.payload for f in frames], [b"g"])

    def test_truncated_frame_waits_for_more(self):
        raw = encode_frame(Command.PROMPT, b"hold on")
        decoder = FrameDecoder()
        self.assertEqual(decoder.feed(raw[:-1]), [])
        frames = decoder.feed(raw[-1:])
        self.assertEqual(frames[0].payload, b"hold on")

    def test_payload_too_long_rejected(self):
        with self.assertRaises(FrameError):
            encode_frame(Command.PROMPT, b"x" * 256)


if __name__ == "__main__":
    unittest.main()