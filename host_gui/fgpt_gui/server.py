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
                 params, warnings) -> None:
        super().__init__(address, handler)
        self.session = session
        self.config = config
        self.params = params
        self.warnings = warnings

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
            "port": getattr(self.config, "port", None),
            "board_params_source": self.params.source,
            "warnings": self.warnings,
            "params": params_dict,
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
        elif path == "/api/ports":
            from .transports.serial_transport import SerialTransport
            ports = SerialTransport.list_ports()
            self._send_json({"ok": True, "ports": ports})
        elif path == "/favicon.ico":
            self.send_error(404)
        else:
            self.send_error(404, "not found")

    # ── POST ──
    def do_POST(self):  # noqa: N802 - stdlib signature
        path = self.path.split("?", 1)[0]
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
        elif path == "/api/settings":
            engine = str(body.get("engine", ""))
            port = str(body.get("port", ""))
            if engine:
                session = self.server.session
                current_is_serial = getattr(session.transport, "name", "") == "serial"

                if engine == "fpga":
                    # Switch to serial (FPGA) transport
                    if not port:
                        self._send_json({"ok": False, "error": "No COM port specified"}, status=400)
                        return
                    try:
                        from .transports.serial_transport import SerialTransport
                        baud = self.server.params.baud_rate
                        new_transport = SerialTransport(port=port, baud=baud)
                        session.swap_transport(new_transport)
                        self.server.config.slm_engine = "fpga"
                        self.server.config.transport = "serial"
                        self.server.config.port = port
                    except Exception as exc:
                        self._send_json({"ok": False, "error": f"FPGA connect failed: {exc}"}, status=500)
                        return
                else:
                    # Switch to SLM transport (smollm, tinystories, etc.)
                    if current_is_serial:
                        # Need to swap back to SLM transport
                        from .transports.slm_transport import SlmTransport
                        new_transport = SlmTransport(
                            engine=engine,
                            gen_tokens=self.server.config.gen_tokens or 160,
                            max_seq_len=1024,
                            vocab_size=self.server.params.vocab_size,
                            temperature=getattr(self.server.config, "slm_temp", 0.7),
                            top_k=getattr(self.server.config, "slm_top_k", 40),
                        )
                        session.swap_transport(new_transport)
                        self.server.config.transport = "slm"
                        self.server.config.port = None
                    else:
                        # Already on SLM, just change the engine
                        session.transport.engine = engine
                        import threading
                        threading.Thread(target=session.transport._ensure_model_loaded, daemon=True).start()
                    self.server.config.slm_engine = engine

                # Broadcast updated config via SSE
                session._emit({"type": "hello", "config": self.server.config_payload(), "state": session.status_event()})
            self._send_json({"ok": True})
        else:
            self.send_error(404, "not found")

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
                  warnings) -> GuiHTTPServer:
    return GuiHTTPServer(
        (config.http_host, config.http_port), GuiRequestHandler,
        session=session, config=config, params=params, warnings=warnings,
    )


def build_from_config(config: GuiConfig):
    """Resolve config into ``(server, session, params, warnings)``."""
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
    server = create_server(config, session, params, warnings)
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