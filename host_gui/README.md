# fpGPT host GUI

A browser UI with two tabs for the DE1-SoC. The **LLM Chat** tab streams
generated characters from the FPGA over UART. The **Food Photo** tab uploads a
picture and runs the bundled TFLite hot-dog classifier on the host.

This folder is self-contained and implements the plan in
[`../docs/frontend_gui_plan.md`](../docs/frontend_gui_plan.md).

## Quick start (no hardware)

The host bridge uses only the Python standard library, so the chat tab runs
immediately with a mock engine:

```bash
# from the repository root
python3 host_gui/run.py                 # mock / echo replies
python3 host_gui/run.py --mock-style ngram   # replies from data/sample.txt
```

Then open <http://127.0.0.1:8765/>.

`echo` cycles the upper-cased prompt to the reply length; `ngram` trains a tiny
character n-gram on `data/sample.txt` and samples a deterministic continuation,
so the UI shows plausible streaming text without a board.

### Food photo tab (optional dependencies)

The classifier needs the TFLite runtime and Pillow, and reads
`Models/hotdog_model.tflite` by default:

```bash
pip install -r host_gui/requirements.txt
python3 host_gui/run.py --food-model Models/hotdog_model.tflite
```

Without those packages the chat tab still works; the Food Photo tab reports the
classifier as unavailable.

## Quick start (real DE1-SoC)

Wire a 3.3 V USB-TTL adapter to `GPIO_0` as described in
[`../fpga/PROGRAMMING.md`](../fpga/PROGRAMMING.md), program the `.sof`, then:

```bash
python3 host_gui/run.py --transport serial --serial-port /dev/ttyUSB0
python3 host_gui/run.py --list-ports          # find the adapter
```

The GUI discovers model geometry and the reply length from
`fpga/board_params.vh` (or `build/rtl/board_params.vh`), so it does not need to
be reconfigured when the model changes. Override with `--board-params PATH`,
`--gen-tokens N`, `--baud N`, `--timeout S`.

### Link modes

- `--mode legacy` (default): matches the **current bitstream**. The host sends
  `prompt + "\n"`, then reads exactly `GEN_TOKENS` characters. No framing/status.
- `--mode framed`: the proposed length-prefixed protocol in
  `fgpt_gui/protocol.py`. This needs the RTL milestone in the plan
  (`uart_protocol.v` + TX backpressure) and will not work with today's
  bitstream. It is implemented and unit-tested host-side so the firmware can be
  brought up against a frozen wire format.

## How it fits together

```
web/ (chat + food tabs)
   |  SSE /api/events  +  POST /api/prompt, /api/reset
   |  POST /api/classify (image bytes)
fgpt_gui/server.py        stdlib HTTP + Server-Sent Events
   |                         |
fgpt_gui/session.py       generation state machine, input filtering, timeouts
   |                         |
fgpt_gui/transports/      serial (pyserial) | mock (echo / ngram)
   |                    fgpt_gui/food_classifier.py  TFLite hot-dog model (host)
DE1-SoC UART (115200 8N1)
```

### Intentional deviation from the plan

The plan recommended FastAPI + WebSockets. The environment this was built in has
no `pip`, so the server is implemented with the standard library and uses
**Server-Sent Events** for the streaming reply instead of a WebSocket. The
session event schema (`status`, `token`, `reply_end`, `error`, `info`) is
unchanged, so replacing the SSE layer with a WebSocket/FastAPI transport later
does not touch `session.py`, `protocol.py`, or the transports.

## API

| Method | Path | Body | Returns |
| --- | --- | --- | --- |
| GET | `/` | – | chat UI |
| GET | `/static/<file>` | – | assets |
| GET | `/api/config` | – | resolved config + board params |
| GET | `/api/events` | – | SSE stream of session events |
| POST | `/api/prompt` | `{"text": "..."}` | `{"ok": bool, "error": ...}` |
| POST | `/api/reset` | `{}` | `{"ok": true}` |
| POST | `/api/classify` | raw image bytes (`Content-Type: image/*`) | `{"ok": bool, "label": ..., "probability": ...}` |

## Tests

No board, no third-party packages required:

```bash
# from the repository root
python3 -m unittest discover -s host_gui/tests -t host_gui -v
```

Covered: `board_params.vh` parsing, frame codec (round-trip, streaming,
corruption, resync), legacy and framed session behaviour (streaming, busy
rejection, timeout, unsupported characters), and the HTTP/SSE server end to end.

## Limitations (inherited from the hardware)

- Replies are a fixed number of bytes (`GEN_TOKENS`); the board has no runtime
  length control and no output backpressure today.
- No cancel/interrupt: the FPGA finishes its burst. `Reset` only clears host
  state (framed mode also sends a `RESET` frame).
- The model is character-level and lightly trained; set expectations
  accordingly.
- `--mode framed` is host-ready but has no matching RTL yet.