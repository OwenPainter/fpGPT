from __future__ import annotations

import json
import threading
import time
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from fgpt_gui.config import GuiConfig
from fgpt_gui.server import build_from_config


class ServerTests(unittest.TestCase):
    def setUp(self):
        config = GuiConfig(transport="mock", mode="legacy", mock_style="echo",
                           gen_tokens=6, timeout_s=3.0, http_port=0)
        self.server, self.session, _params, _warnings = build_from_config(config)
        self.session.start()
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.session.stop()

    def test_config_endpoint(self):
        with urlopen(self.base + "/api/config", timeout=5) as response:
            payload = json.load(response)
        self.assertEqual(payload["transport"], "mock")
        self.assertEqual(payload["gen_tokens"], 6)
        self.assertIn("params", payload)

    def test_index_and_static(self):
        with urlopen(self.base + "/", timeout=5) as response:
            html = response.read().decode("utf-8")
        self.assertIn("fpGPT", html)
        with urlopen(self.base + "/static/app.js", timeout=5) as response:
            js = response.read().decode("utf-8")
        self.assertIn("EventSource", js)

    def test_prompt_and_streamed_reply(self):
        events = []
        lock = threading.Lock()

        def reader():
            try:
                with urlopen(self.base + "/api/events", timeout=8) as response:
                    for line in response:
                        if not line.startswith(b"data: "):
                            continue
                        event = json.loads(line[len(b"data: "):])
                        with lock:
                            events.append(event)
                        if event.get("type") == "reply_end":
                            return
            except Exception:
                pass

        reader_thread = threading.Thread(target=reader, daemon=True)
        reader_thread.start()
        time.sleep(0.25)  # let the SSE client subscribe

        request = Request(
            self.base + "/api/prompt",
            data=json.dumps({"text": "AB"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=5) as response:
            result = json.load(response)
        self.assertTrue(result["ok"])

        reader_thread.join(timeout=8)
        types = [event["type"] for event in events]
        self.assertIn("token", types)
        self.assertIn("reply_end", types)
        tokens = [e for e in events if e["type"] == "token"]
        self.assertEqual(len(tokens), 6)

    def test_reset_endpoint(self):
        request = Request(self.base + "/api/reset", data=b"{}",
                          headers={"Content-Type": "application/json"},
                          method="POST")
        with urlopen(request, timeout=5) as response:
            self.assertTrue(json.load(response)["ok"])

    def test_rejects_unsupported_prompt(self):
        # board_params.vh has VOCAB_SIZE=64, so ASCII 32..92 only; lowercase
        # letters are outside the hardware tokenizer's range.
        request = Request(
            self.base + "/api/prompt",
            data=json.dumps({"text": "lowercase"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with self.assertRaises(HTTPError) as ctx:
            urlopen(request, timeout=5)
        self.assertEqual(ctx.exception.code, 409)


if __name__ == "__main__":
    unittest.main()