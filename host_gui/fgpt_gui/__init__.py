"""fpGPT host GUI.

A dependency-free host bridge that turns the board's UART character stream into
a browser chat UI. See ``host_gui/README.md``.

Design summary (mirrors docs/frontend_gui_plan.md):

    web/  ->  server.py (HTTP + SSE)  ->  session.py  ->  transports/  ->  board

The protocol/session/transport layers are pure standard library so they can be
unit-tested and exercised with mock data even when no FPGA or UART adapter is
attached.
"""

from __future__ import annotations

__version__ = "0.1.0"
PROTOCOL_VERSION = 1