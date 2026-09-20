"""Dependency-free host server: static files + JSON + Server-Sent Events.

The browser UI is a single page served from ``fgpt_gui/web``. Because the
standard library has no WebSocket support and this environment has no third-party
packages, streaming replies use Server-Sent Events (``GET /api/events``) and
prompts are submitted with plain ``POST`` requests. The event schema is exactly
the one in the plan, so swapping in a WebSocket/FastAPI transport later needs no
changes to ``session.py``.

Endpoints
---------
``GET  /``             chat UI
``GET  /static/<file>`` CSS/JS assets
``GET  /api/config``   resolved config + board params
``GET  /api/events``   SSE stream of session events
``POST /api/prompt``   ``{"text": "..."}`` -> ``{"ok": bool, "error": ...}``
``POST /api/reset``    best-effort clear
``POST /api/classify`` raw image bytes -> ``{"ok": bool, "label": ...}``
"""

from __future__ import annotations

import json
import queue
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from . import PROTOCOL_VERSION, __version__
from .board_params import load_board_params
from .config import GuiConfig
from .food_classifier import (
    MAX_IMAGE_BYTES,
    ClassifierUnavailable,
    FoodClassifier,
)
from .session import ChatSession
from .transports import build_transport

WEB_DIR = Path(__file__).resolve().parent / "web"

_CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".svg": "image/svg+xml",
    ".ico": "image/x-icon",
    ".json": "application/json",
}


class GuiHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address, handler, session: ChatSession, config: GuiConfig,
                 params, warnings, classifier: FoodClassifier | None = None) -> None:
        super().__init__(address, handler)
        self.session = session
        self.config = config
        self.params = params
        self.warnings = warnings
        self.classifier = classifier

    def config_payload(self) -> dict:
        params_dict = self.params.to_dict()
        if self.config.transport == "slm":
            params_dict["char_min"] = 32
            params_dict["char_max"] = 126
        return {
            "app": "fpGPT host GUI",
            "version": __version__,
            "protocol_version": PROTOCOL_VERSION,
            "transport": self.config.transport,
            "mode": self.config.mode,
            "gen_tokens": self.session.gen_tokens,
            "timeout_s": self.config.timeout_s,
            "mock_style": self.config.mock_style,
            "slm_engine": getattr(self.config, "slm_engine", "smollm"),
            "board_params_source": self.params.source,
            "warnings": self.warnings,
            "params": params_dict,
            "food": self.classifier.status() if self.classifier else {"available": False},
        }


class GuiRequestHandler(BaseHTTPRequestHandler):
    server_version = "fpGPT-host-gui"

    # Keep logs terse but useful.
    def log_message(self, fmt, *args):  # noqa: A003 - stdlib signature
        print(f"[host_gui] {self.address_string()} {fmt % args}")

    # ── GET ──
    def do_GET(self):  # noqa: N802 - stdlib signature
        path = self.path.split("?", 1)[0]
        if path == "/":
            self._send_file(WEB_DIR / "index.html")
        elif path.startswith("/static/"):
            self._send_static(path[len("/static/"):])
        elif path == "/api/config":
            self._send_json(self.server.config_payload())
        elif path == "/api/events":
            self._serve_events()
        elif path == "/favicon.ico":
            self.send_error(404)
        else:
            self.send_error(404, "not found")

    # ── POST ──
    def do_POST(self):  # noqa: N802 - stdlib signature
        path = self.path.split("?", 1)[0]
        if path == "/api/classify":
            self._handle_classify()
            return
        try:
            body = self._read_json()
        except ValueError as exc:
            self._send_json({"ok": False, "error": str(exc)}, status=400)
            return

        if path == "/api/prompt":
            text = str(body.get("text", ""))
            ok, error = self.server.session.submit(text)
            self._send_json({"ok": ok, "error": error}, status=200 if ok else 409)
        elif path == "/api/reset":
            self.server.session.reset()
            self._send_json({"ok": True})
        elif path == "/api/shutdown":
            self._send_json({"ok": True, "message": "Server shutting down..."})
            import threading
            threading.Thread(target=self.server.shutdown, daemon=True).start()
        else:
            self.send_error(404, "not found")

    # ── helpers ──
    def _handle_classify(self) -> None:
        classifier = self.server.classifier
        if classifier is None:
            self._send_json({"ok": False, "error": "food classifier disabled"},
                            status=503)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        if length <= 0:
            self._send_json({"ok": False, "error": "no image data"}, status=400)
            return
        if length > MAX_IMAGE_BYTES:
            self._send_json({"ok": False, "error": "image too large"}, status=413)
            return
        image_bytes = self.rfile.read(length)
        try:
            result = classifier.predict(image_bytes)
        except ClassifierUnavailable as exc:
            self._send_json({"ok": False, "error": str(exc)}, status=503)
        except ValueError as exc:
            self._send_json({"ok": False, "error": str(exc)}, status=400)
        else:
            self._send_json(result)

    def _read_json(self) -> dict:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        raw = self.rfile.read(length) if length else b"{}"
        if not raw:
            return {}
        try:
            data = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("invalid JSON body") from exc
        if not isinstance(data, dict):
            raise ValueError("JSON body must be an object")
        return data

    def _send_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path, status: int = 200) -> None:
        try:
            body = path.read_bytes()
        except OSError:
            self.send_error(404, "not found")
            return
        content_type = _CONTENT_TYPES.get(path.suffix, "application/octet-stream")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_static(self, relative: str) -> None:
        candidate = (WEB_DIR / relative).resolve()
        try:
            candidate.relative_to(WEB_DIR.resolve())
        except ValueError:
            self.send_error(403, "forbidden")
            return
        if not candidate.is_file():
            self.send_error(404, "not found")
            return
        self._send_file(candidate)

    def _serve_events(self) -> None:
        session = self.server.session
        q = session.subscribe()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache, no-store")
        self.send_header("Connection", "keep-alive")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()

        try:
            self._sse_send({"type": "hello", "config": self.server.config_payload(),
                            "state": session.status_event()})
            while True:
                try:
                    event = q.get(timeout=15.0)
                except queue.Empty:
                    self.wfile.write(b": keep-alive\n\n")
                    self.wfile.flush()
                    continue
                self._sse_send(event)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            session.unsubscribe(q)

    def _sse_send(self, event: dict) -> None:
        payload = json.dumps(event).encode("utf-8")
        self.wfile.write(b"data: " + payload + b"\n\n")
        self.wfile.flush()


def create_server(config: GuiConfig, session: ChatSession, params,
                  warnings, classifier: FoodClassifier | None = None) -> GuiHTTPServer:
    return GuiHTTPServer(
        (config.http_host, config.http_port), GuiRequestHandler,
        session=session, config=config, params=params, warnings=warnings,
        classifier=classifier,
    )


def build_from_config(config: GuiConfig, classifier: FoodClassifier | None = None):
    """Resolve config into ``(server, session, params, warnings)``.

    ``classifier`` may be injected (e.g. a stub in tests); otherwise a real
    :class:`FoodClassifier` is built from ``config.food_model``.
    """
    config.validate()
    params, warnings = load_board_params(config.board_params)
    transport = build_transport(config, params)
    session = ChatSession(
        transport=transport,
        params=params,
        mode=config.mode,
        gen_tokens=config.gen_tokens,
        reply_timeout=config.timeout_s,
    )
    if classifier is None:
        classifier = FoodClassifier(config.food_model)
    server = create_server(config, session, params, warnings, classifier)
    return server, session, params, warnings


def serve(config: GuiConfig) -> None:
    """Start the host bridge and block until interrupted."""
    server, session, params, warnings = build_from_config(config)
    host, port = server.server_address[0], server.server_address[1]
    print(f"fpGPT host GUI v{__version__} (protocol v{PROTOCOL_VERSION})")
    print(f"  transport : {config.transport}"
          + (f" ({config.port} @ {config.baud or params.baud_rate})"
             if config.transport == "serial" else f" ({config.mock_style})"))
    print(f"  link mode : {config.mode}")
    print(f"  model     : d_model={params.d_model} layers={params.num_layers} "
          f"vocab={params.vocab_size} seq={params.max_seq_len}")
    print(f"  gen bytes : {session.gen_tokens} per prompt")
    if params.source:
        print(f"  params    : {params.source}")
    food = server.classifier.status() if server.classifier else {"available": False}
    if food.get("available"):
        print(f"  food tab  : {food['model']} (TFLite)")
    else:
        print(f"  food tab  : unavailable ({food.get('error', 'disabled')})")
    for warning in warnings:
        print(f"  WARNING   : {warning}")
    print(f"  open      : http://{host}:{port}/")
    session.start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[host_gui] shutting down")
    finally:
        server.shutdown()
        server.server_close()
        session.stop()