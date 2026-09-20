from __future__ import annotations

import json
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from fgpt_gui.config import GuiConfig
from fgpt_gui.food_classifier import FoodClassifier
from fgpt_gui.server import build_from_config


class StubClassifier:
    """Deterministic stand-in so the endpoint tests need no TFLite runtime."""

    def __init__(self, result=None, error=None):
        self.result = result or {
            "ok": True,
            "label": "Hot dog",
            "is_hotdog": True,
            "probability": 0.87,
            "confidence": 0.87,
            "threshold": 0.5,
            "input_size": [150, 150],
            "model": "stub.tflite",
        }
        self.error = error

    def status(self):
        return {"available": True, "model": "stub.tflite", "threshold": 0.5}

    def predict(self, image_bytes):
        if self.error:
            raise self.error
        return dict(self.result)


class FoodEndpointTests(unittest.TestCase):
    def _start(self, classifier):
        config = GuiConfig(transport="mock", mode="legacy", mock_style="echo",
                           gen_tokens=6, timeout_s=3.0, http_port=0)
        self.server, self.session, _params, _warnings = build_from_config(
            config, classifier=classifier)
        self.session.start()
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.session.stop()

    def test_config_reports_food_status(self):
        self._start(StubClassifier())
        with urlopen(self.base + "/api/config", timeout=5) as response:
            payload = json.load(response)
        self.assertTrue(payload["food"]["available"])
        self.assertEqual(payload["food"]["model"], "stub.tflite")

    def test_classify_returns_prediction(self):
        self._start(StubClassifier())
        request = Request(self.base + "/api/classify", data=b"fake-jpeg-bytes",
                          headers={"Content-Type": "image/jpeg"}, method="POST")
        with urlopen(request, timeout=5) as response:
            result = json.load(response)
        self.assertTrue(result["ok"])
        self.assertEqual(result["label"], "Hot dog")

    def test_classify_rejects_empty_body(self):
        self._start(StubClassifier())
        request = Request(self.base + "/api/classify", data=b"",
                          headers={"Content-Type": "image/jpeg"}, method="POST")
        with self.assertRaises(HTTPError) as ctx:
            urlopen(request, timeout=5)
        self.assertEqual(ctx.exception.code, 400)

    def test_classify_surfaces_decode_error(self):
        self._start(StubClassifier(error=ValueError("could not decode image")))
        request = Request(self.base + "/api/classify", data=b"not-an-image",
                          headers={"Content-Type": "image/jpeg"}, method="POST")
        with self.assertRaises(HTTPError) as ctx:
            urlopen(request, timeout=5)
        self.assertEqual(ctx.exception.code, 400)


class ClassifierStatusTests(unittest.TestCase):
    def test_missing_model_is_unavailable(self):
        classifier = FoodClassifier(model_path="definitely/not/here.tflite")
        status = classifier.status()
        self.assertFalse(status["available"])
        self.assertFalse(status["model_exists"])

    def test_bundled_model_reports_available(self):
        classifier = FoodClassifier()
        if not classifier.model_path.is_file():
            self.skipTest("bundled Models/hotdog_model.tflite not present")
        status = classifier.status()
        # Available depends on the optional TFLite runtime being installed.
        if status["runtime_installed"]:
            self.assertTrue(status["available"])
            self.assertEqual(status["threshold"], 0.35)
            self.assertEqual(status["not_hotdog_min_confidence"], 0.65)


class ThresholdRuleTests(unittest.TestCase):
    def test_not_hotdog_only_when_above_65_percent_confidence(self):
        from unittest.mock import MagicMock
        import numpy as np

        classifier = FoodClassifier()
        classifier._size = (150, 150)
        classifier._input = {"index": 0}
        classifier._output = {"index": 1}

        # Mock image bytes
        from PIL import Image
        import io
        img = Image.new("RGB", (150, 150), color="red")
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        fake_bytes = buf.getvalue()

        def test_score(raw_score: float):
            mock_interp = MagicMock()
            mock_interp.get_tensor.return_value = np.array([[raw_score]], dtype=np.float32)
            classifier._interpreter = mock_interp
            return classifier.predict(fake_bytes)

        # 1. Very strong not-hotdog (score 0.10 => 90% not-hotdog, which is > 65%)
        res = test_score(0.10)
        self.assertFalse(res["is_hotdog"])
        self.assertEqual(res["label"], "Not a hot dog.")
        self.assertAlmostEqual(res["confidence"], 0.90, places=2)

        # 2. Borderline not-hotdog (score 0.34 => 66% not-hotdog, which is > 65%)
        res = test_score(0.34)
        self.assertFalse(res["is_hotdog"])
        self.assertEqual(res["label"], "Not a hot dog.")
        self.assertAlmostEqual(res["confidence"], 0.66, places=2)

        # 3. Weak not-hotdog (score 0.40 => 60% not-hotdog, which is NOT > 65%) -> MUST be Hot dog!
        res = test_score(0.40)
        self.assertTrue(res["is_hotdog"])
        self.assertEqual(res["label"], "Hot dog!")
        self.assertAlmostEqual(res["confidence"], 0.40, places=2)

        # 4. Score 0.48 (52% not-hotdog, NOT > 65%) -> MUST be Hot dog!
        res = test_score(0.48)
        self.assertTrue(res["is_hotdog"])
        self.assertEqual(res["label"], "Hot dog!")

        # 5. Definite hot dog (score 0.85) -> Hot dog!
        res = test_score(0.85)
        self.assertTrue(res["is_hotdog"])
        self.assertEqual(res["label"], "Hot dog!")
        self.assertAlmostEqual(res["confidence"], 0.85, places=2)


if __name__ == "__main__":
    unittest.main()

