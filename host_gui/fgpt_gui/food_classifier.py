"""Hot-dog image classifier backed by a TFLite model.

This is the optional "food photo" half of the host GUI. It is deliberately
isolated from the chat path so the FPGA bridge keeps working with the standard
library alone: the heavy dependencies (``ai-edge-litert`` for inference and
``Pillow`` for image decoding) are imported lazily and only when a picture is
actually classified.

The model is the one trained in the fpGPT "seefood" Colab: a small CNN with a
``(150, 150, 3)`` float32 input scaled to ``[0, 1]`` and a single sigmoid output.
The notebook's own ``predict_image`` treats ``score > 0.5`` as a hot dog, so
that is the convention used here.
"""

from __future__ import annotations

import importlib.util
import io
import threading
from pathlib import Path

# Repository root is two levels up from ``host_gui/fgpt_gui/``.
DEFAULT_MODEL = Path(__file__).resolve().parents[2] / "Models" / "hotdog_model.tflite"

HOTDOG_THRESHOLD = 0.5
MAX_IMAGE_BYTES = 12 * 1024 * 1024

HOTDOG_LABEL = "Hot dog!"
NOT_HOTDOG_LABEL = "Not a hot dog."


class ClassifierUnavailable(RuntimeError):
    """Raised when the TFLite runtime or model file is missing."""


class FoodClassifier:
    """Lazy wrapper around the TFLite interpreter.

    Construction is cheap; the model is only loaded on the first prediction.
    ``status`` performs the same dependency check without loading the model so
    ``/api/config`` stays fast.
    """

    def __init__(self, model_path: str | None = None) -> None:
        self.model_path = Path(model_path) if model_path else DEFAULT_MODEL
        self._interpreter = None
        self._input = None
        self._output = None
        self._size: tuple[int, int] | None = None
        self._lock = threading.Lock()

    # ── capability reporting ──
    def status(self) -> dict:
        runtime_installed = importlib.util.find_spec("ai_edge_litert") is not None
        model_exists = self.model_path.is_file()
        available = runtime_installed and model_exists
        info = {
            "available": available,
            "model": self.model_path.name,
            "model_path": str(self.model_path),
            "model_exists": model_exists,
            "runtime_installed": runtime_installed,
            "threshold": HOTDOG_THRESHOLD,
        }
        if self._size is not None:
            info["input_size"] = list(self._size)
        if not available:
            if not model_exists:
                info["error"] = f"model not found: {self.model_path}"
            else:
                info["error"] = (
                    "ai-edge-litert is not installed "
                    "(pip install ai-edge-litert Pillow)"
                )
        return info

    # ── inference ──
    def _load(self) -> None:
        if self._interpreter is not None:
            return
        try:
            from ai_edge_litert.interpreter import Interpreter
        except ImportError as exc:  # pragma: no cover - exercised via status()
            raise ClassifierUnavailable(
                "ai-edge-litert is not installed "
                "(pip install ai-edge-litert Pillow)"
            ) from exc
        if not self.model_path.is_file():
            raise ClassifierUnavailable(f"model not found: {self.model_path}")

        interpreter = Interpreter(model_path=str(self.model_path))
        interpreter.allocate_tensors()
        self._interpreter = interpreter
        self._input = interpreter.get_input_details()[0]
        self._output = interpreter.get_output_details()[0]
        shape = self._input["shape"]
        self._size = (int(shape[1]), int(shape[2]))

    def predict(self, image_bytes: bytes) -> dict:
        """Classify raw image bytes and return a JSON-friendly result."""
        if not image_bytes:
            raise ValueError("empty image")
        if len(image_bytes) > MAX_IMAGE_BYTES:
            raise ValueError("image too large")

        try:
            import numpy as np
            from PIL import Image
        except ImportError as exc:  # pragma: no cover - exercised via status()
            raise ClassifierUnavailable(
                "Pillow/numpy are not installed "
                "(pip install ai-edge-litert Pillow)"
            ) from exc

        import time
        t0 = time.perf_counter()
        with self._lock:
            self._load()
            try:
                image = Image.open(io.BytesIO(image_bytes))
                image.load()
            except Exception as exc:
                raise ValueError("could not decode image") from exc

            image = image.convert("RGB").resize(self._size)
            array = np.asarray(image, dtype=np.float32) / 255.0
            array = array[None, ...]

            self._interpreter.set_tensor(self._input["index"], array)
            self._interpreter.invoke()
            score = float(np.reshape(self._interpreter.get_tensor(
                self._output["index"]), -1)[0])

        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
        is_hotdog = score >= HOTDOG_THRESHOLD
        return {
            "ok": True,
            "label": HOTDOG_LABEL if is_hotdog else NOT_HOTDOG_LABEL,
            "is_hotdog": is_hotdog,
            "probability": score,
            "confidence": score if is_hotdog else 1.0 - score,
            "threshold": HOTDOG_THRESHOLD,
            "input_size": list(self._size),
            "model": self.model_path.name,
            "inference_ms": elapsed_ms,
        }
