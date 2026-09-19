"""In-process mock transport: develop the GUI with no board attached.

It emulates the board's observable behaviour closely enough for the session to
be exercised end to end:

* it collects prompt bytes until a CR/LF, exactly like
  ``generation_controller.v``;
* it then streams exactly ``gen_tokens`` characters back, one at a time, with a
  small delay so the UI shows real streaming;
* it accepts only characters the tokenizer supports and maps unsupported ones
  to ``'.'`` the way the RTL does.

Two reply styles are available:

``echo``
    Deterministic: cycles the upper-cased prompt text to ``gen_tokens`` chars.
``ngram``
    A tiny character n-gram model over a corpus (default ``data/sample.txt``),
    seeded deterministically from the prompt, so replies look like plausible
    (if low-quality) language. Falls back to ``echo`` if no corpus exists.
"""

from __future__ import annotations

import random
import threading
import time
from collections import defaultdict
from collections import deque
from pathlib import Path

from .base import Transport, TransportError

DEFAULT_CORPUS = Path(__file__).resolve().parents[3] / "data" / "sample.txt"


class MockTransport(Transport):
    name = "mock"

    def __init__(
        self,
        gen_tokens: int = 16,
        max_seq_len: int = 64,
        vocab_size: int = 64,
        style: str = "echo",
        corpus=None,
        byte_delay: float = 0.01,
    ) -> None:
        self.gen_tokens = int(gen_tokens)
        self.max_seq_len = int(max_seq_len)
        self.vocab_size = int(vocab_size)
        self.style = style
        self.byte_delay = byte_delay
        self._corpus_path = Path(corpus) if corpus else DEFAULT_CORPUS

        self._lock = threading.Lock()
        self._pending: deque[tuple[float, int]] = deque()
        self._prompt = bytearray()
        self._open = False
        self._ngram = None

        if style == "ngram":
            self._ngram = _CharNgram.from_file(self._corpus_path,
                                                high=self.char_max)
            if self._ngram is None:
                self.style = "echo"

    @property
    def char_max(self) -> int:
        return 32 + self.vocab_size - 3 - 1

    # ── lifecycle ──
    def open(self) -> None:
        self._open = True

    def close(self) -> None:
        self._open = False
        with self._lock:
            self._pending.clear()
            self._prompt.clear()

    # ── transmit ──
    def send(self, data: bytes) -> None:
        if not self._open:
            raise TransportError("mock transport is not open")
        reply_chars = None
        with self._lock:
            for byte in data:
                if byte in (0x0A, 0x0D):
                    if self._prompt:
                        reply_chars = self._build_reply(bytes(self._prompt))
                        self._prompt.clear()
                elif len(self._prompt) < self.max_seq_len:
                    self._prompt.append(byte)

            if reply_chars:
                now = time.monotonic()
                for index, char in enumerate(reply_chars):
                    self._pending.append((now + index * self.byte_delay, char))

    def read(self, timeout: float) -> bytes:
        deadline = time.monotonic() + max(0.0, timeout)
        while True:
            with self._lock:
                now = time.monotonic()
                ready = []
                while self._pending and self._pending[0][0] <= now:
                    ready.append(self._pending.popleft()[1])
            if ready:
                return bytes(ready)
            if time.monotonic() >= deadline:
                return b""
            time.sleep(min(0.005, max(0.0, deadline - time.monotonic())))

    def reset(self) -> None:
        with self._lock:
            self._pending.clear()
            self._prompt.clear()

    # ── reply generation ──
    def _build_reply(self, prompt: bytes) -> list[int]:
        text = prompt.decode("latin-1", errors="replace").strip()
        if self.style == "ngram" and self._ngram is not None:
            text = self._ngram.generate(text, self.gen_tokens)
        else:
            text = self._echo_text(text)
        return [self._map_char(ch) for ch in text[: self.gen_tokens]]

    def _echo_text(self, text: str) -> str:
        base = (text.upper() if text else "MOCK ENGINE")
        return (base * (self.gen_tokens // max(1, len(base)) + 1))[: self.gen_tokens]

    def _map_char(self, ch: str) -> int:
        code = ord(ch)
        low = 32
        high = 32 + self.vocab_size - 3 - 1
        if low <= code <= high:
            return code
        return ord(".")


class _CharNgram:
    """Minimal character trigram model with deterministic sampling."""

    ORDER = 3

    def __init__(self, text: str, high: int = 126) -> None:
        # Keep only characters the hardware tokenizer can represent, and
        # upper-case so a lower-case corpus still yields visible text under a
        # small (e.g. VOCAB_SIZE=64) contract.
        low = 32
        text = "".join(
            c for c in text.upper() if low <= ord(c) <= high
        )
        self._orders = [dict() for _ in range(self.ORDER + 1)]
        self._text = text
        padded = " " * self.ORDER + text + " "
        for n in range(1, self.ORDER + 1):
            table = self._orders[n]
            for i in range(len(padded) - n):
                context = padded[i:i + n]
                table.setdefault(context, defaultdict(int))
                table[context][padded[i + n]] += 1
        self._fallback = " "
        if text:
            counts: dict = {}
            for ch in text:
                counts[ch] = counts.get(ch, 0) + 1
            self._fallback = max(counts, key=counts.get)

    def generate(self, seed: str, length: int) -> str:
        context = (seed or " ")[-self.ORDER :]
        context = " " * (self.ORDER - len(context)) + context
        rng = random.Random(seed or "\x00")
        out: list[str] = []
        for _ in range(length):
            char, context = self._advance(context, rng)
            out.append(char)
        return "".join(out)

    def _advance(self, context: str, rng: random.Random):
        """Return ``(next_char, next_context)``, backing off then reseeding."""
        for n in range(self.ORDER, 0, -1):
            table = self._orders[n].get(context[-n:])
            if table:
                chars = list(table)
                weights = [table[c] for c in chars]
                nxt = rng.choices(chars, weights=weights, k=1)[0]
                return nxt, (context + nxt)[-self.ORDER :]

        # No continuation for this context: jump to a random corpus position so
        # the mock keeps emitting plausible word fragments instead of padding.
        if self._text:
            pos = rng.randrange(len(self._text))
            nxt = self._text[pos]
            prefix = self._text[max(0, pos - self.ORDER + 1):pos]
            fresh = (prefix + nxt)
            fresh = (" " * (self.ORDER - len(fresh)) + fresh)[-self.ORDER :]
            return nxt, fresh
        return self._fallback, (context + self._fallback)[-self.ORDER :]

    @classmethod
    def from_file(cls, path: Path, high: int = 126):
        try:
            text = Path(path).read_text(encoding="utf-8", errors="replace")
        except OSError:
            return None
        if not text.strip():
            return None
        return cls(text, high=high)