from __future__ import annotations

import queue
import time
import unittest

from fgpt_gui.board_params import BoardParams
from fgpt_gui.protocol import Command, DoneReason, encode_frame
from fgpt_gui.session import ChatSession, SessionState
from fgpt_gui.transports.base import Transport
from fgpt_gui.transports.mock_transport import MockTransport

PARAMS = BoardParams(vocab_size=98, max_seq_len=64, gen_tokens=8)


def collect(session, q, until="reply_end", timeout=5.0):
    events = []
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            event = q.get(timeout=deadline - time.monotonic())
        except queue.Empty:
            break
        events.append(event)
        if event.get("type") == until:
            break
    return events


class SilentTransport(Transport):
    name = "silent"

    def open(self):
        pass

    def close(self):
        pass

    def send(self, data):
        pass

    def read(self, timeout):
        time.sleep(min(timeout, 0.01))
        return b""


class ScriptedTransport(Transport):
    """Emits a fixed byte script, two bytes at a time, once prompted."""

    name = "script"

    def __init__(self, script: bytes):
        self.script = script
        self._pending = bytearray()
        self.sent = bytearray()

    def open(self):
        pass

    def close(self):
        pass

    def send(self, data):
        self.sent.extend(data)
        self._pending.extend(self.script)

    def read(self, timeout):
        if not self._pending:
            time.sleep(min(timeout, 0.02))
            return b""
        chunk = bytes(self._pending[:2])
        del self._pending[:2]
        return chunk


class LegacySessionTests(unittest.TestCase):
    def test_oversized_prompt_is_rejected_before_transmission(self):
        transport = ScriptedTransport(b"12345678")
        session = ChatSession(transport, PARAMS)
        ok, error = session.submit("A" * 57)
        self.assertFalse(ok)
        self.assertIn("56 characters", error)
        self.assertEqual(transport.sent, b"")
        self.assertEqual(session.state, SessionState.IDLE)
        ok, error = session.submit("A" * 56)
        self.assertTrue(ok, error)
        self.assertEqual(transport.sent, b"A" * 56 + b"\n")

    def make_session(self, **kwargs):
        transport = MockTransport(gen_tokens=8, vocab_size=98, byte_delay=0.005)
        return ChatSession(transport, PARAMS, mode="legacy", **kwargs)

    def test_streams_full_reply(self):
        session = self.make_session()
        q = session.subscribe()
        session.start()
        try:
            ok, error = session.submit("hello")
            self.assertTrue(ok, error)
            events = collect(session, q)
        finally:
            session.stop()

        tokens = [e for e in events if e["type"] == "token"]
        self.assertEqual(len(tokens), 8)
        reply = [e for e in events if e["type"] == "reply_end"]
        self.assertEqual(len(reply), 1)
        self.assertEqual(reply[0]["reason"], "complete")
        self.assertEqual(reply[0]["text"], "HELLOHEL")
        self.assertEqual(session.state, SessionState.IDLE)

    def test_rejects_prompt_while_busy(self):
        session = self.make_session()
        session.start()
        try:
            ok, _ = session.submit("first")
            self.assertTrue(ok)
            ok2, error = session.submit("second")
            self.assertFalse(ok2)
            self.assertIn("busy", error)
        finally:
            session.stop()

    def test_sanitize_drops_unsupported(self):
        session = self.make_session()
        clean, removed = session.sanitize("hi\u0001there\t!")
        self.assertEqual(clean, "hithere!")
        self.assertEqual(len(removed), 2)  # \x01 and tab

    def test_empty_prompt_rejected(self):
        session = self.make_session()
        ok, error = session.submit("\u0000\u0001")
        self.assertFalse(ok)
        self.assertIn("empty", error)

    def test_timeout_when_board_is_silent(self):
        session = ChatSession(SilentTransport(), PARAMS, mode="legacy",
                              reply_timeout=0.2)
        q = session.subscribe()
        session.start()
        try:
            session.submit("hello")
            events = collect(session, q)
        finally:
            session.stop()
        reply = [e for e in events if e["type"] == "reply_end"]
        self.assertEqual(reply[0]["reason"], "timeout")


class FramedSessionTests(unittest.TestCase):
    def test_framed_tokens_and_done(self):
        script = (
            encode_frame(Command.TOKEN, b"h")
            + encode_frame(Command.TOKEN, b"i")
            + encode_frame(Command.DONE, bytes([int(DoneReason.OK)]))
        )
        session = ChatSession(ScriptedTransport(script), PARAMS, mode="framed",
                              gen_tokens=8)
        q = session.subscribe()
        session.start()
        try:
            ok, error = session.submit("go")
            self.assertTrue(ok, error)
            events = collect(session, q)
        finally:
            session.stop()

        self.assertEqual([e["char"] for e in events if e["type"] == "token"],
                         ["h", "i"])
        reply = [e for e in events if e["type"] == "reply_end"]
        self.assertEqual(reply[0]["text"], "hi")
        self.assertEqual(reply[0]["reason"], "ok")

    def test_framed_error_event(self):
        script = encode_frame(Command.ERROR, b"\x01")
        session = ChatSession(ScriptedTransport(script), PARAMS, mode="framed")
        q = session.subscribe()
        session.start()
        try:
            session.submit("go")
            events = collect(session, q, until="error")
        finally:
            session.stop()
        self.assertTrue(any(e["type"] == "error" for e in events))


if __name__ == "__main__":
    unittest.main()
