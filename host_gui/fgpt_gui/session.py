"""Chat session: turns a transport byte stream into UI events.

The session owns the generation state machine and is deliberately independent
of HTTP. It supports two link modes:

``legacy``
    Works with today's bitstream: send ``prompt + "\\n"``, then read exactly
    ``gen_tokens`` characters. No framing, no status — the reply length is known
    from ``board_params.vh``.
``framed``
    The proposed protocol in ``protocol.py``: PROMPT + GENERATE out, TOKEN /
    DONE frames back. Requires the RTL milestone described in the plan.

Events are plain dicts with a ``type`` key so they serialise straight to JSON:
``status``, ``info``, ``token``, ``reply_end``, ``error``.
"""

from __future__ import annotations

import queue
import threading
import time
from enum import Enum

from . import PROTOCOL_VERSION
from .protocol import (
    Command,
    DoneReason,
    FrameDecoder,
    encode_frame,
)

DEFAULT_REPLY_TIMEOUT = 5.0


class SessionState(str, Enum):
    IDLE = "idle"
    COLLECTING = "collecting"
    ERROR = "error"


class ChatSession:
    def __init__(self, transport, params, mode: str = "legacy",
                 gen_tokens: int | None = None,
                 reply_timeout: float = DEFAULT_REPLY_TIMEOUT) -> None:
        self.transport = transport
        self.params = params
        self.mode = mode
        if getattr(transport, "name", "") == "slm":
            self.gen_tokens = int(gen_tokens or getattr(transport, "gen_tokens", 256))
        else:
            self.gen_tokens = int(gen_tokens or params.gen_tokens)
        self.reply_timeout = float(reply_timeout)

        self._state = SessionState.IDLE
        self._lock = threading.Lock()
        self._subscribers: list[queue.Queue] = []
        self._reply = bytearray()
        self._last_activity = 0.0
        self._decoder = FrameDecoder()
        self._reader_thread: threading.Thread | None = None
        self._running = threading.Event()
        self._error: str | None = None

    # ── lifecycle ──
    def start(self) -> None:
        self.transport.open()
        self._running.set()
        self._reader_thread = threading.Thread(
            target=self._reader_loop, name="fgpt-reader", daemon=True
        )
        self._reader_thread.start()
        self._emit(self.status_event())
        self._emit({"type": "info", "params": self.params.to_dict(),
                    "mode": self.mode, "gen_tokens": self.gen_tokens,
                    "protocol_version": PROTOCOL_VERSION})

    def stop(self) -> None:
        self._running.clear()
        thread = self._reader_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)
        self._reader_thread = None
        try:
            self.transport.close()
        except Exception:
            pass

    def __enter__(self) -> "ChatSession":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    # ── subscriber hub (SSE fanout) ──
    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=128)
        with self._lock:
            self._subscribers.append(q)
        q.put_nowait(self.status_event())
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self._lock:
            if q in self._subscribers:
                self._subscribers.remove(q)

    def _emit(self, event: dict) -> None:
        with self._lock:
            subscribers = list(self._subscribers)
        for q in subscribers:
            try:
                q.put_nowait(event)
            except queue.Full:
                pass

    # ── state ──
    @property
    def state(self) -> SessionState:
        return self._state

    @property
    def busy(self) -> bool:
        return self._state == SessionState.COLLECTING

    def status_event(self) -> dict:
        return {
            "type": "status",
            "state": self._state.value,
            "mode": self.mode,
            "busy": self.busy,
            "error": self._error,
        }

    # ── input ──
    def sanitize(self, text: str) -> tuple[str, list[str]]:
        """Drop characters the hardware tokenizer cannot represent."""
        clean = []
        removed = []
        is_slm = getattr(self.transport, "name", "") == "slm"
        for ch in text:
            code = ord(ch)
            if ch in "\r\n":
                continue
            if is_slm:
                # Full printable ASCII + standard text characters supported by SLM
                if 32 <= code <= 126 or code > 127:
                    clean.append(ch)
                else:
                    removed.append(ch)
            else:
                if self.params.char_min <= code <= self.params.char_max:
                    clean.append(ch)
                else:
                    removed.append(ch)
        return "".join(clean), removed

    def submit(self, text: str) -> tuple[bool, str | None]:
        """Queue a prompt. Returns ``(ok, error)``."""
        if self.busy:
            return False, "engine busy; wait for the current reply"

        clean, removed = self.sanitize(text)
        if not clean:
            return False, "prompt is empty after removing unsupported characters"

        with self._lock:
            self._reply.clear()
            self._error = None
            if self.mode == "framed":
                self._decoder.reset()
            self._last_activity = time.monotonic()
            self._state = SessionState.COLLECTING

        if self.mode == "legacy":
            payload = clean.encode("utf-8", errors="replace") + b"\n"
        else:
            payload = encode_frame(Command.PROMPT, clean.encode("utf-8", "replace"))
            payload += encode_frame(
                Command.GENERATE, bytes([self.gen_tokens & 0xFF, (self.gen_tokens >> 8) & 0xFF])
            )

        self._emit({"type": "status", "state": "collecting", "mode": self.mode,
                    "busy": True, "removed": len(removed)})
        try:
            self.transport.send(payload)
        except Exception as exc:
            self._fail(str(exc))
            return False, str(exc)
        return True, None

    def reset(self) -> None:
        """Best-effort clear. The RTL only stops on a real reset/KEY[0]."""
        if self.mode == "framed":
            try:
                self.transport.send(encode_frame(Command.RESET))
            except Exception as exc:
                self._fail(str(exc))
                return
        with self._lock:
            self._reply.clear()
            self._state = SessionState.IDLE
            self._error = None
        self._emit({"type": "status", "state": "idle", "mode": self.mode,
                    "busy": False})

    # ── reader loop ──
    def _reader_loop(self) -> None:
        while self._running.is_set():
            try:
                data = self.transport.read(0.1)
            except Exception as exc:
                if self._running.is_set():
                    self._fail(str(exc))
                return
            if data:
                self._last_activity = time.monotonic()
                try:
                    self._handle_bytes(data)
                except Exception as exc:  # defensive: never kill the thread
                    self._fail(str(exc))
                    return
            elif self.busy and (time.monotonic() - self._last_activity
                                > self.reply_timeout):
                self._finish("timeout")

    def _handle_bytes(self, data: bytes) -> None:
        if self.mode == "legacy":
            if not self.busy:
                return  # unsolicited bytes are ignored on the legacy link
            for byte in data:
                # 0x04 is EOT (End of Transmission) marker emitted by SLM when generation finishes
                if getattr(self.transport, "name", "") == "slm" and byte == 0x04:
                    self._finish("complete")
                    return
                self._append_reply(byte)
                if len(self._reply) >= self.gen_tokens:
                    self._finish("complete")
                    return
            return

        # Framed: always parse, even when idle (READY/INFO can arrive any time).
        for frame in self._decoder.feed(data):
            self._handle_frame(frame)

    def _handle_frame(self, frame) -> None:
        command = frame.cmd
        if command is Command.TOKEN and frame.payload:
            self._append_reply(frame.payload[0])
        elif command is Command.DONE:
            reason = frame.payload[0] if frame.payload else int(DoneReason.OK)
            self._finish(_reason_name(reason))
        elif command is Command.ERROR:
            self._fail(f"board error {frame.payload.hex()}")
        elif command is Command.INFO_RESP:
            self._emit({"type": "info", "raw": frame.payload.hex()})
        elif command is Command.READY:
            self._emit({"type": "ready",
                        "protocol_version": frame.payload[0] if frame.payload else 0})

    def _append_reply(self, byte: int) -> None:
        self._reply.append(byte & 0xFF)
        char = chr(byte) if (32 <= byte < 127 or byte in (10, 13, 9)) else "."
        self._emit({"type": "token", "char": char, "raw": byte & 0xFF})

    def _finish(self, reason: str) -> None:
        with self._lock:
            if self._state != SessionState.COLLECTING:
                return
            text = self._reply.decode("utf-8", errors="replace")
            self._state = SessionState.IDLE
        self._emit({"type": "reply_end", "reason": reason, "text": text,
                    "tokens": len(self._reply)})
        self._emit(self.status_event())

    def _fail(self, message: str) -> None:
        with self._lock:
            self._error = message
            self._state = SessionState.ERROR
        self._emit({"type": "error", "message": message})
        self._emit(self.status_event())


def _reason_name(reason: int) -> str:
    try:
        return DoneReason(reason).name.lower()
    except ValueError:
        return f"reason_{reason}"