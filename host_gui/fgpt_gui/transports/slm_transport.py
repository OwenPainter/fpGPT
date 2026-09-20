"""In-process Small Language Model (SLM) transport.

Runs local, offline autoregressive generation on CPU with zero API keys.
Supports:
  - "microgpt" (default): Loads the repository's PyTorch model
    (``checkpoints/micro_gpt.pt``) and generates tokens using the custom
    character-level architecture.
  - "smollm": HuggingFace SmolLM2-135M-Instruct (if ``transformers`` is
    installed) for natural conversational dialogue.

Tokens are streamed incrementally through ``read()`` with an optional pacing
delay so the browser UI displays smooth, live generation.
"""

from __future__ import annotations

import sys
import threading
import time
from collections import deque
from pathlib import Path

from .base import Transport, TransportError

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_CHECKPOINT = REPO_ROOT / "checkpoints" / "micro_gpt.pt"


class SlmTransport(Transport):
    name = "slm"

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        engine: str = "microgpt",
        gen_tokens: int = 64,
        max_seq_len: int = 64,
        vocab_size: int = 64,
        temperature: float = 0.8,
        top_k: int = 4,
        byte_delay: float = 0.02,
        hf_model: str = "HuggingFaceTB/SmolLM2-135M-Instruct",
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else DEFAULT_CHECKPOINT
        self.engine = engine.lower()
        self.gen_tokens = int(gen_tokens)
        self.max_seq_len = int(max_seq_len)
        self.vocab_size = int(vocab_size)
        self.temperature = float(temperature)
        self.top_k = int(top_k)
        self.byte_delay = float(byte_delay)
        self.hf_model = hf_model

        self._open = False
        self._lock = threading.Lock()
        self._pending: deque[tuple[float, int]] = deque()
        self._prompt = bytearray()

        # Lazy model references
        self._micro_model = None
        self._micro_tokenizer = None
        self._hf_pipeline = None

    # ── lifecycle ──
    def open(self) -> None:
        self._open = True
        self._ensure_model_loaded()

    def close(self) -> None:
        self._open = False
        with self._lock:
            self._pending.clear()
            self._prompt.clear()

    def reset(self) -> None:
        with self._lock:
            self._pending.clear()
            self._prompt.clear()

    # ── model loading ──
    def _ensure_model_loaded(self) -> None:
        if self.engine == "smollm":
            if self._hf_pipeline is not None:
                return
            try:
                from transformers import pipeline
                self._hf_pipeline = pipeline(
                    "text-generation",
                    model=self.hf_model,
                    device="cpu",
                )
            except Exception:
                # Fallback to microgpt if transformers/model is missing
                self.engine = "microgpt"

        if self.engine == "microgpt" and self._micro_model is None:
            try:
                import torch
                from model.micro_gpt import MicroGPT
                from model.tokenizer import CharTokenizer
            except ImportError as exc:
                self._load_error = f"PyTorch is not installed in this Python ({sys.executable}): {exc}"
                print(f"[slm_transport] WARNING: {self._load_error}. Falling back to mock echo.")
                self.engine = "fallback"
                return

            if not self.checkpoint_path.is_file():
                raise TransportError(f"MicroGPT checkpoint not found: {self.checkpoint_path}")

            ckpt = torch.load(str(self.checkpoint_path), map_location="cpu", weights_only=False)
            config = ckpt.get("config", {
                "vocab_size": self.vocab_size,
                "max_seq_len": self.max_seq_len,
                "d_model": 64,
                "num_heads": 4,
                "num_layers": 4,
                "d_ff": 256,
                "dropout": 0.0,
            })
            model = MicroGPT(config)
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()

            self._micro_model = model
            vocab_size = config.get("vocab_size", self.vocab_size)
            self._micro_tokenizer = CharTokenizer(vocab_size=vocab_size)

    # ── I/O ──
    def send(self, data: bytes) -> None:
        if not self._open:
            raise TransportError("SLM transport is not open")

        with self._lock:
            for byte in data:
                if byte in (0x0A, 0x0D):
                    if self._prompt:
                        prompt_bytes = bytes(self._prompt)
                        self._prompt.clear()
                        # Run generation in background so send() doesn't block the caller
                        threading.Thread(
                            target=self._generate_reply,
                            args=(prompt_bytes,),
                            daemon=True,
                        ).start()
                elif len(self._prompt) < self.max_seq_len:
                    self._prompt.append(byte)

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

    # ── text generation ──
    def _generate_reply(self, prompt_bytes: bytes) -> None:
        text = prompt_bytes.decode("utf-8", errors="replace").strip()
        reply_chars = []

        if self.engine == "smollm" and self._hf_pipeline is not None:
            try:
                out = self._hf_pipeline(
                    text,
                    max_new_tokens=self.gen_tokens,
                    temperature=self.temperature,
                    do_sample=True,
                    top_k=self.top_k,
                )
                generated = out[0]["generated_text"][len(text):]
                reply_chars = [ord(c) for c in generated[:self.gen_tokens] if ord(c) < 256]
            except Exception:
                reply_chars = []

        if not reply_chars and self._micro_model is not None:
            import torch
            try:
                # Tokenize prompt using character tokenizer
                tok_ids = self._micro_tokenizer.encode(text)
                if not tok_ids:
                    tok_ids = [1]  # SOS / default start

                x = torch.tensor([tok_ids[-self.max_seq_len:]], dtype=torch.long)
                with torch.no_grad():
                    gen_ids = self._micro_model.generate(
                        x,
                        max_new_tokens=self.gen_tokens,
                        temperature=self.temperature,
                        top_k=self.top_k,
                    )
                new_tokens = gen_ids[0, len(tok_ids):].tolist()
                decoded_text = self._micro_tokenizer.decode(new_tokens)
                reply_chars = [ord(c) for c in decoded_text[:self.gen_tokens] if ord(c) < 256]
            except Exception as exc:
                err_msg = f" [SLM error: {exc}] "
                reply_chars = [ord(c) for c in err_msg]

        if not reply_chars:
            fallback_text = (
                f"[PyTorch unavailable in this python executable. Run with: python launchWebsite.py] Echo: {text.upper()}"
            )
            reply_chars = [ord(c) for c in fallback_text[: self.gen_tokens] if ord(c) < 256]

        # Queue generated characters with pacing timestamps for smooth streaming
        now = time.monotonic()
        with self._lock:
            for idx, char_code in enumerate(reply_chars):
                self._pending.append((now + idx * self.byte_delay, char_code))
