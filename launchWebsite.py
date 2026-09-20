#!/usr/bin/env python3
"""Host the fpGPT web UI locally.

Serves the browser app (LLM Chat + Food Photo tabs) from ``host_gui``. With no
extra arguments it uses the mock transport, so it runs without an FPGA attached.

    python launchWebsite.py                       # mock/echo on 127.0.0.1:8765
    python launchWebsite.py --open                # also open the browser
    python launchWebsite.py --mock-style ngram    # replies from data/sample.txt
    python launchWebsite.py --transport serial --serial-port COM3
    python launchWebsite.py --host 0.0.0.0 --http-port 9000

Everything after the script name is forwarded to the host GUI CLI, so
``python launchWebsite.py --help`` lists the available options.
"""

from __future__ import annotations

import sys
import threading
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "host_gui"))

from fgpt_gui.cli import build_parser, config_from_args, main as gui_main  # noqa: E402


def _food_dependency_note() -> str | None:
    import importlib.util

    missing = [name for name, module in (("ai-edge-litert", "ai_edge_litert"),
                                         ("Pillow", "PIL"))
               if importlib.util.find_spec(module) is None]
    if missing:
        return ("Food Photo tab needs " + ", ".join(missing)
                + " (pip install -r host_gui/requirements.txt); "
                "the chat tab still works.")
    return None


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    open_browser = "--open" in argv
    argv = [arg for arg in argv if arg != "--open"]

    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list_ports:
        return gui_main(argv)

    config = config_from_args(args)
    try:
        config.validate()
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    host = "127.0.0.1" if config.http_host in ("", "0.0.0.0") else config.http_host
    url = f"http://{host}:{config.http_port}/"

    if open_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()

    print(f"fpGPT host GUI -> {url}")
    note = _food_dependency_note()
    if note:
        print(f"  note: {note}")

    return gui_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
