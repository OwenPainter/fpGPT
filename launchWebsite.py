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


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    # ── Stop running instance if requested ──
    if "--stop" in argv:
        import urllib.request
        port = 8765
        for i, a in enumerate(argv):
            if a in ("--http-port", "--port") and i + 1 < len(argv):
                try:
                    port = int(argv[i + 1])
                except ValueError:
                    pass
        url = f"http://127.0.0.1:{port}/api/shutdown"
        try:
            req = urllib.request.Request(
                url, data=b"{}", headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=3.0) as resp:
                print(f"[launchWebsite] Successfully stopped server at http://127.0.0.1:{port}/")
                return 0
        except Exception as exc:
            print(f"[launchWebsite] Could not reach server at {url}: {exc}")
            return 1

    no_open = "--no-open" in argv
    argv = [arg for arg in argv if arg not in ("--open", "--no-open")]

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

    if not no_open:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()

    print(f"==================================================")
    print(f"  fpGPT Web Interface -> {url}")
    print(f"  LLM Chat   : In-Process SLM ({config.slm_engine} on CPU)")
    print(f"  Controls   : Press Ctrl+C or run 'python launchWebsite.py --stop'")
    print(f"==================================================")

    return gui_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
