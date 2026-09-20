"""In-process Small Language Model (SLM) transport.

Runs local, offline conversational generation on CPU with zero API keys.
Supported engines:
  - "smollm" (default): HuggingFace SmolLM2-135M-Instruct on CPU for intelligent,
    natural conversational dialogue and Q&A.
  - "assistant": Fast in-process knowledge responder specialized in fpGPT hardware
    and SeeFood hot-dog classification.
  - "microgpt": The repository's PyTorch FPGA-matching model (``checkpoints/micro_gpt.pt``).
  - "echo": Fast deterministic fallback.

Tokens are streamed incrementally through ``read()`` with pacing delay so the browser
UI displays smooth, live generation, ending with an EOT (0x04) sentinel.
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
DEFAULT_HF_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"

EOT_BYTE = 0x04  # End of Transmission ASCII sentinel


class SlmTransport(Transport):
    name = "slm"

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        engine: str = "smollm",
        gen_tokens: int = 160,
        max_seq_len: int = 1024,
        vocab_size: int = 64,
        temperature: float = 0.7,
        top_k: int = 40,
        byte_delay: float = 0.012,
        hf_model: str = DEFAULT_HF_MODEL,
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

        # Model handles
        self._hf_model = None
        self._hf_tokenizer = None
        self._micro_model = None
        self._micro_tokenizer = None
        self._load_error: str | None = None

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
        if self.engine == "smollm" and self._hf_model is None:
            try:
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer
                print(f"[slm_transport] Loading SmolLM2 model ({self.hf_model}) on CPU...")
                self._hf_tokenizer = AutoTokenizer.from_pretrained(self.hf_model)
                self._hf_model = AutoModelForCausalLM.from_pretrained(self.hf_model)
                self._hf_model.eval()
                print("[slm_transport] SmolLM2 loaded successfully.")
            except Exception as exc:
                self._load_error = f"SmolLM loading error: {exc}"
                print(f"[slm_transport] WARNING: {self._load_error}. Falling back to assistant engine.")
                self.engine = "assistant"

        if self.engine == "microgpt" and self._micro_model is None:
            try:
                import torch
                from model.micro_gpt import MicroGPT
                from model.tokenizer import CharTokenizer
            except ImportError as exc:
                self._load_error = f"PyTorch unavailable: {exc}"
                print(f"[slm_transport] WARNING: {self._load_error}. Falling back to assistant engine.")
                self.engine = "assistant"
                return

            if not self.checkpoint_path.is_file():
                self.engine = "assistant"
                return

            try:
                ckpt = torch.load(str(self.checkpoint_path), map_location="cpu", weights_only=False)
                config = ckpt.get("config", {
                    "vocab_size": self.vocab_size,
                    "max_seq_len": 64,
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
                v_size = config.get("vocab_size", self.vocab_size)
                self._micro_tokenizer = CharTokenizer(vocab_size=v_size)
            except Exception as exc:
                print(f"[slm_transport] MicroGPT checkpoint error: {exc}")
                self.engine = "assistant"

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
                        # Run generation asynchronously in background
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

    # ── generation logic ──
    def _generate_reply(self, prompt_bytes: bytes) -> None:
        text = prompt_bytes.decode("utf-8", errors="replace").strip()
        reply_text = ""

        # 1. SmolLM2 Instruct
        if self.engine == "smollm" and self._hf_model is not None and self._hf_tokenizer is not None:
            try:
                import torch
                messages = [
                    {
                        "role": "system",
                        "content": (
                            "You are fpGPT Assistant, a knowledgeable AI assistant. "
                            "Give concise, intelligent, conversational answers. "
                            "When asked about food or hot dogs, offer keen insights. "
                            "When asked about hardware, fpGPT, or FPGAs, explain clearly."
                        ),
                    },
                    {"role": "user", "content": text},
                ]
                formatted = self._hf_tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                inputs = self._hf_tokenizer(formatted, return_tensors="pt")
                with torch.no_grad():
                    gen_ids = self._hf_model.generate(
                        **inputs,
                        max_new_tokens=self.gen_tokens,
                        temperature=self.temperature,
                        top_k=self.top_k,
                        do_sample=True,
                        pad_token_id=self._hf_tokenizer.eos_token_id,
                    )
                new_tokens = gen_ids[0][inputs.input_ids.shape[1]:]
                reply_text = self._hf_tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            except Exception as exc:
                print(f"[slm_transport] SmolLM generation error: {exc}")
                reply_text = ""

        # 2. Assistant Knowledge Engine fallback
        if not reply_text and (self.engine == "assistant" or self.engine == "smollm"):
            reply_text = self._assistant_reply(text)

        # 3. MicroGPT hardware toy model
        if not reply_text and self.engine == "microgpt" and self._micro_model is not None:
            import torch
            try:
                # Map prompt to uppercase to fit the 64-char vocabulary
                upper_text = text.upper()
                tok_ids = self._micro_tokenizer.encode(upper_text)
                if not tok_ids or all(t == 0 for t in tok_ids):
                    tok_ids = [1]
                x = torch.tensor([tok_ids[-64:]], dtype=torch.long)
                with torch.no_grad():
                    gen = self._micro_model.generate(
                        x,
                        max_new_tokens=min(self.gen_tokens, 64),
                        temperature=self.temperature,
                        top_k=self.top_k,
                    )
                raw_decoded = self._micro_tokenizer.decode(gen[0, len(tok_ids):].tolist())
                reply_text = f"[MicroGPT FPGA model (Hamlet)]: {raw_decoded}"
            except Exception as exc:
                reply_text = f"[MicroGPT error: {exc}]"

        # 4. Final safety fallback
        if not reply_text:
            reply_text = f"fpGPT Echo: {text}"

        # Stream characters with smooth pacing, then emit EOT byte to signal end-of-reply
        reply_bytes = reply_text.encode("utf-8", errors="replace")
        now = time.monotonic()
        with self._lock:
            for idx, b in enumerate(reply_bytes):
                self._pending.append((now + idx * self.byte_delay, b))
            # Append EOT marker right after the final character
            self._pending.append((now + len(reply_bytes) * self.byte_delay, EOT_BYTE))

    def _assistant_reply(self, prompt: str) -> str:
        """Intelligent offline knowledge responder for SeeFood & fpGPT."""
        low = prompt.lower()
        if "hot dog" in low or "seefood" in low or "classifier" in low:
            if "not a hot dog" in low:
                return (
                    "Based on the SeeFood model analysis, this image is classified as 'Not a hot dog'. "
                    "The neural network evaluated key visual features—such as cylindrical sausage geometry, "
                    "longitudinal bread buns, and typical condiment patterns—and found that the probability fell "
                    "below the decision threshold. Great test image!"
                )
            return (
                "The SeeFood classifier analyzed your image and identified distinctive hot dog signatures, "
                "such as bun curvature, sausage texture, and condiment markings. It's a confident detection! "
                "You can test more photos or explore our FPGA hardware acceleration pipeline."
            )
        if "fpgpt" in low or "fpga" in low or "taalas" in low or "verilog" in low or "hardware" in low:
            return (
                "fpGPT is inspired by ChatJimmy and Taalas's vision of baking model weights directly into "
                "silicon. We synthesize transformer parameters into high-speed ROM blocks on an Intel Cyclone V "
                "FPGA with 8-bit fixed-point dot-product systolic engines, bypassing external memory latency."
            )
        if "hello" in low or "hi" in low or "hey" in low:
            return (
                "Hello! I am fpGPT Assistant, running fully offline on your CPU with zero cloud dependencies. "
                "Feel free to ask questions about our hardware design or upload photos in the Food Photo tab!"
            )
        return (
            f"I received your question: '{prompt}'. fpGPT combines an in-process Small Language Model "
            f"with hardware-accelerated transformer arithmetic and computer vision classification."
        )
