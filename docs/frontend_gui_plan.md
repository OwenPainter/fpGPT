# fpGPT Frontend GUI + Text-Chat Plan

Status: **plan only — no code written yet.**
Goal: a frontend GUI with a chat-style text interface that lets a human type a
prompt and watch the on-FPGA Micro-GPT stream generated characters back, with
no LLM running on the host.

This document is the implementation contract for that work. It is scoped to the
host/GUI side plus the minimum firmware changes needed to make the link robust.

---

## 1. What already exists (constraints we must design around)

The chip-side datapath is implemented and simulated; the host side does not exist.

| Fact | Source | Consequence for the GUI |
| --- | --- | --- |
| UART is 115200 8N1, no flow control | `fpga/PROGRAMMING.md:98` | One byte ≈ 87 µs; engine is slower, so tokens trickle |
| Prompt = raw bytes, CR or LF starts decoding | `fpga/generation_controller.v:179-198` | No framing, no prompt length, no header |
| Output = exactly `GEN_TOKENS` bytes, then silence | `fpga/generation_controller.v:243-247` | Host must know the count out-of-band; there is no start/end marker |
| `GEN_TOKENS` is a compile-time parameter, default 16 | `fpga/board_params.vh:37` | Runtime "max tokens" control is impossible without RTL changes |
| No TX backpressure; `uart_tx` can drop bytes if `start` is asserted while busy | `README.md:308`, `fpga/fpga_top.v:154` | Long/fast output can corrupt; protocol should pace or the RTL must add buffering |
| Tokenizer is char-level: specials 0–2, then printable ASCII 32..`32+VOCAB-4` | `model/tokenizer.py:7-14`, `fpga/generation_controller.v:127-145` | GUI must filter input to the supported range; output may include `.` for specials |
| Sampling is LFSR top-k only, no temperature | `README.md:312` | No temperature control to expose yet |
| Model becomes ready only after a newline; generation is non-interruptible | `fpga/generation_controller.v:147-252` | Need a "busy/streaming" state in the UI; no cancel |
| `board_params.vh` carries geometry + baud + gen count | `fpga/board_params.vh` | Host can auto-discover config instead of hard-coding |
| Default model is small and lightly trained | `README.md:1-16,351` | Set expectations in the UI; quality is not the point |

Hardware caveat: the UART is on `GPIO_0[0:1]` and needs an external 3.3 V
USB-TTL adapter (`fpga/PROGRAMMING.md:86-98`). Serious GUI work must therefore be
possible with **no board attached**.

---

## 2. Goals and non-goals

### Goals
1. A browser-based chat UI: transcript, prompt box, streaming assistant reply.
2. A local host bridge (Python) that owns the serial port and exposes a clean
   API to the UI (`WebSocket` + `HTTP`).
3. A **mock/simulated backend** so the GUI is fully developable and testable
   without an FPGA or a serial adapter.
4. A **legacy mode** that speaks the current bitstream behavior (prompt + newline,
   read `GEN_TOKENS` bytes) so the GUI works with today's hardware unmodified.
5. A **framed mode** with a documented, versioned UART protocol (commands, token
   stream, done/status) for the next bitstream revision.
6. Configuration discovered from `board_params.vh`, not duplicated in the GUI.
7. Hardware-abstracted transport so mock, legacy and framed share one code path.
8. Tests that run in CI without hardware.

### Non-goals (for this phase)
- Training, compiling, or programming the FPGA from the GUI.
- Temperature / advanced sampling (not in RTL yet).
- Token-level (BPE) tokenizers; the model is character-level.
- Remote/multi-user hosting; bind to loopback only.
- Cancel/stop of an in-flight generation (RTL has no interrupt).

---

## 3. Recommended architecture

```
┌────────────────────────────────────────────────────────────────────┐
│ Browser (web/)                                                     │
│   index.html + app.js + style.css                                  │
│   chat transcript · prompt box · streamed reply · status LEDs       │
└───────────────▲────────────────────────────────────────────────────┘
                │ WebSocket (JSON events) + HTTP (static + REST)
┌───────────────┴────────────────────────────────────────────────────┐
│ Host bridge (Python, host/)                                        │
│   server.py     FastAPI + uvicorn, loopback only                    │
│   session.py    chat state machine, serializes generations          │
│   protocol.py   framed codec, framing constants, checksum           │
│   board_params.py  parse board_params.vh                            │
│   transports/   base · serial · mock · (optional) icarus-sim        │
└───────────────▲────────────────────────────────────────────────────┘
                │ pyserial (115200 8N1)        …or… no I/O (mock)
┌───────────────┴────────────────────────────────────────────────────┐
│ FPGA over UART  —  fpga_top.v / generation_controller.v             │
│   legacy: prompt+"\n" → GEN_TOKENS bytes                            │
│   framed: STX len cmd payload cksum ETX                             │
└────────────────────────────────────────────────────────────────────┘
```

### Stack decision (recommended)
- **Backend:** Python 3.12, `fastapi` + `uvicorn`, `pyserial`, `pydantic`.
  Python-first matches the repo; WebSockets give token-by-token streaming.
- **Frontend:** plain HTML/CSS/vanilla JS ES modules, no build step. Served as
  static files by FastAPI. Keeps the repo dependency-light and reviewable; a
  framework can be layered later without changing the backend contract.
- **No new heavy deps in the core path.** `fastapi`, `uvicorn[standard]` and
  `pyserial` go in a separate `host/requirements.txt` so `make test` for the
  core model/compiler is unaffected.

Alternatives considered and rejected for now: Streamlit/Gradio (fast to build but
weak streaming/state control and heavy deps), Electron/PySide6 (desktop packaging
burden), Node frontend build chain (adds toolchain without benefit).

---

## 4. Proposed file layout

```
host/
├── __init__.py
├── cli.py                 # `python -m host.cli --port /dev/ttyUSB0 --mode framed`
├── config.py              # dataclass: transport, port, baud, mode, gen_tokens
├── board_params.py        # parse FPGPT_* macros from board_params.vh
├── protocol.py            # frame encode/decode, checksums, command enums
├── session.py             # ChatSession: prompt → streamed reply, busy state
├── server.py              # FastAPI app, WebSocket endpoint, static mount
├── requirements.txt       # fastapi, uvicorn, pyserial
└── transports/
    ├── __init__.py
    ├── base.py            # Transport ABC: write(bytes), read loop → events
    ├── serial_transport.py
    └── mock_transport.py  # echo + integer-reference mocks

web/
├── index.html
├── app.js
└── style.css

tests/
├── test_board_params.py
├── test_protocol.py
├── test_session.py
└── test_server.py

docs/
├── frontend_gui_plan.md        # this file
└── uart_protocol.md            # written in Phase 5
```

---

## 5. The two link modes

### 5.1 Legacy mode (works with today's bitstream)
- Host sends the prompt bytes followed by `\n` (0x0A).
- Host reads exactly `GEN_TOKENS` bytes (default from `board_params.vh`) with an
  inter-byte timeout, then emits a `done` event.
- No header/status, no runtime token count, no error signal.
- Known limitation: if the RTL ever streams faster than UART, bytes can be lost.
  GUI must surface this as a warning and recommend framed mode.

### 5.2 Framed mode (Phase 5, requires RTL changes)
Length-prefixed binary framing so printable payloads and framing bytes coexist:

```
Frame:  STX(0x02) | LEN(1) | CMD(1) | PAYLOAD[LEN] | CKSUM(1) | ETX(0x03)
CKSUM = 8-bit XOR of LEN, CMD and PAYLOAD
```

Host → FPGA:

| CMD | Name | Payload | Meaning |
| --- | --- | --- | --- |
| 0x01 | PING | – | Liveness check |
| 0x02 | PROMPT | raw bytes | Append to prompt buffer (no trigger) |
| 0x03 | GENERATE | `n_tokens:u8, flags:u8` | Run n forward passes after current prompt |
| 0x04 | RESET | – | Clear prompt buffer / return to collect |
| 0x05 | INFO | – | Request geometry/version |

FPGA → host:

| CMD | Name | Payload | Meaning |
| --- | --- | --- | --- |
| 0x81 | READY | `proto_ver:u8` | Reset complete |
| 0x82 | TOKEN | `byte:u8` | One generated character |
| 0x83 | DONE | `reason:u8, cycles:u4` | Burst complete |
| 0x84 | INFO | `d_model,layers,vocab,seq_len` | Geometry |
| 0x85 | ERROR | `code:u8` | Framing/overflow/busy |

RTL work implied (tracked separately, not in the first GUI milestone):
- A small UART command parser + response framer in `fpga/` (new `uart_protocol.v`).
- A TX FIFO / backpressure flag so `uart_tx` never drops generated bytes
  (addresses `README.md:308`).
- Runtime token count fed into `generation_controller` instead of `GEN_TOKENS`.

The host codec is written first and unit-tested independently of RTL, so the
firmware can be brought up against a frozen, tested wire format.

---

## 6. Component design

### 6.1 `board_params.py`
Parse `` `define FPGPT_* `` values from `fpga/board_params.vh` (or
`build/rtl/board_params.vh`) into a dataclass: `VOCAB_SIZE`, `MAX_SEQ_LEN`,
`GEN_TOKENS`, `BAUD_RATE`, `DATA_WIDTH`, geometry. Used to auto-configure
legacy mode and to show model info in the UI. Missing/invalid file → documented
defaults with a visible warning.

### 6.2 `protocol.py`
- `encode_frame(cmd, payload) -> bytes`, `FrameDecoder.feed(byte)` streaming
  decoder tolerant of garbage (resync on STX), checksum validation, max LEN guard.
- Command enums shared with docs. No I/O; pure functions → trivially unit-testable.

### 6.3 `transports/base.py`
Async-friendly interface:
```python
class Transport(Protocol):
    async def open(self) -> None: ...
    async def close(self) -> None: ...
    async def send(self, data: bytes) -> None: ...
    def events(self) -> AsyncIterator[Event]: ...  # RAW_BYTE | TOKEN | DONE | ERROR | STATUS
```
- `serial_transport.py`: `pyserial` opened in a reader thread/executor, pushes
  bytes into an asyncio queue; exposes `reset` via DTR if wired (optional).
- `mock_transport.py`: two flavors.
  - `EchoMock`: deterministic canned/capitalized text; used for UI tests.
  - `ReferenceMock`: uses `model/fixed_point.py` / `model/tokenizer.py` integer
    reference to produce real, if low-quality, text with no hardware. Falls back
    to `EchoMock` if the checkpoint is absent.
- (Optional) `icarus_transport.py`: drives `tests/tb_fpga_top.v`-style simulation
  per generation. High value, high cost — explicitly optional/last.

### 6.4 `session.py`
`ChatSession` owns one transport and serializes generations:
- State: `IDLE`, `COLLECTING` (receiving `GEN_TOKENS`/TOKEN frames), `ERROR`.
- `submit(prompt: str)`: validate/filter chars to the tokenizer range, send
  (legacy: prompt+`\n`; framed: PROMPT+`GENERATE`), set state, start timeout.
- Aggregates streamed bytes into the assistant message; emits UI events.
- Rejects new prompts while busy (display "engine busy; wait for burst").
- Timeout per burst; marks partial output on timeout.
- Keeps a bounded transcript and the last raw byte log for debugging.

### 6.5 `server.py`
- `GET /` serves `web/index.html`; static mount for `web/`.
- `GET /api/config` → mode, board params, connection state.
- `WS /ws` events (JSON):
  - client→server: `{type:"prompt", text}`, `{type:"reset"}`, `{type:"ping"}`
  - server→client: `{type:"status", state, mode}`, `{type:"token", char, raw}`,
    `{type:"reply_end", reason, cycles?}`, `{type:"error", message}`,
    `{type:"info", params}`
- One `ChatSession` per server, one active socket (single-user, loopback).
- CLI (`host/cli.py`): `--transport {serial,mock,echo}`, `--port`, `--baud`,
  `--mode {legacy,framed}`, `--board-params PATH`, `--host/--port` for HTTP.

### 6.6 Frontend `web/`
- Chat transcript (user / assistant bubbles), streaming assistant text with a
  blinking cursor; monospace because output is ASCII.
- Input box: Enter to send, Shift+Enter newline; live char counter vs
  `MAX_SEQ_LEN`; non-printable/unsupported chars highlighted and optionally
  stripped before send with a warning.
- Status bar echoing the board LEDs: connection, reset, PLL-locked
  (unknown over legacy), engine busy, last byte.
- Controls: **Reset/prompt clear**, **connection mode** indicator, **Export
  transcript** (download `.txt`), **copy reply**. Token-count selector only in
  framed mode; disabled with a tooltip in legacy mode.
- Settings drawer: transport, port, baud, mode (`read-only` for board params).
- Clear "prototype / not-a-chatbot" banner so users don't expect ChatGPT quality.

---

## 7. Milestones and acceptance criteria

### Phase 0 — Decisions + scaffolding
- Confirm stack and where docs live; add `host/requirements.txt`.
- `docs/uart_protocol.md` skeleton.
- **Done when:** directory skeleton + dependency file reviewed; no behavior yet.

### Phase 1 — Config + codec (no UI)
- `board_params.py`, `protocol.py` with pytest coverage.
- **Done when:** `pytest tests/test_board_params.py tests/test_protocol.py`
  passes; round-trip + corrupt-frame + resync tests green.

### Phase 2 — Transports + session
- `base.py`, `serial_transport.py`, `mock_transport.py`, `session.py`.
- **Done when:** `test_session.py` runs a full prompt→streamed-reply→`done`
  cycle against `EchoMock` and against a mocked serial byte stream; timeout and
  busy-rejection paths covered.

### Phase 3 — Server + WebSocket
- `server.py`, `cli.py`, event schema.
- **Done when:** `test_server.py` uses FastAPI's TestClient to assert
  `status → token* → reply_end` for a submitted prompt.

### Phase 4 — Frontend chat
- `web/` chat UI wired to `/ws`; manual QA against `echo` and `mock`.
- **Done when:** typing a prompt streams a reply character-by-character in a
  browser with no hardware attached; status/LED bar updates; input filtering works.

### Phase 5 — Hardware legacy mode
- Point CLI at the USB-TTL adapter, read `board_params.vh`, send prompt+`\n`,
  read `GEN_TOKENS` bytes with inter-byte timeout.
- **Done when:** on a programmed DE1-SoC, a typed prompt returns the expected
  number of characters and they appear live in the GUI; `PROGRAMMING.md` gains a
  "host GUI" section.

### Phase 6 — Framed protocol (RTL + host)
- Implement `uart_protocol.v` + TX backpressure/runtime token count in RTL.
- Switch host to framed mode; keep legacy for old bitstreams.
- **Done when:** `test_generation_controller.py`-style RTL test exercises frames
  end-to-end, and the GUI reports cycles/status from `DONE`.

### Phase 7 — Polish, CI, docs
- Add `host/` tests to CI (no hardware; mock transports only).
- README + `docs/uart_protocol.md` + `TODO.md` updated.
- **Done when:** `make test` unchanged/green and a new `make test-host` target
  runs the host suite.

---

## 8. Testing strategy
- **Unit:** codec encode/decode/checksum/resync; board-params parser; char
  filtering vs `CharTokenizer.byte_to_id` semantics.
- **Integration (no hardware):** session over `EchoMock` and over a fake serial
  that replays a captured byte trace; FastAPI WebSocket test.
- **Golden:** reuse/emit a small legacy byte trace so protocol changes can't
  silently break the old path.
- **Hardware smoke (manual, opt-in):** documented runbook step; never in CI.
- Keep the core model/compiler tests untouched; host deps are isolated.

## 9. Risks and open questions
- **Protocol change appetite:** framed mode needs RTL work. *Proposal:* ship
  legacy mode first so the GUI is useful immediately, then framed.
- **Byte loss:** current `uart_tx` has no backpressure (`README.md:308`). Legacy
  mode is only safe if the engine is slower than UART; the GUI must warn and the
  framed RTL milestone must add a FIFO.
- **UI stack:** vanilla JS recommended; if a framework is preferred, the
  WebSocket contract stays fixed.
- **No cancel/reset from host:** `KEY[0]` is physical; the GUI can only clear its
  own prompt. Framed `RESET` is the eventual answer.
- **Generate quality / EOS:** the model may never emit EOS and is lightly
  trained; UI treats replies as fixed-length bursts, not turn-based chat.
- **Checkpoint dependency for `ReferenceMock`:** if `checkpoints/micro_gpt.pt`
  is missing, that mock degrades to `EchoMock`.
- **Platform/serial driver:** assumes a USB-TTL adapter appears as
  `/dev/ttyUSB*` or `COMx`; discovery/selection is a CLI flag, not automatic.

## 10. Explicitly deferred
Runtime sampling controls, model/compile/program buttons, multi-user or network
exposure, tokenizer beyond char-level, generation interrupt, and GUI-driven
Quartus flows. Each has a natural hook (framed protocol, `compile.py`, HTTP API)
when wanted.