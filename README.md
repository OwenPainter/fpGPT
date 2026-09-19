# fpGPT — ML Weights to Silicon Compiler

**Turn trained neural-network weights into synthesizable Verilog RTL that is
intended to run on FPGA fabric with no CPU, OS, or software runtime.**

Inspired by [Taalas/ChatJimmy](https://chatjimmy.ai), which bakes LLM weights
into ASIC silicon. fpGPT targets the same "model-on-silicon" idea using
reconfigurable FPGA fabric and a Micro-GPT small enough to fit in on-chip
block RAM.

> **Status: research prototype — not yet run on hardware.**
> The compiler, the RTL building blocks, a full forward-pass engine, and a
> board-level generation controller all exist and are simulated in Icarus
> Verilog. The design has never been synthesized, fitted, or run on a
> DE1-SoC, and the board ROM is not yet wired to compiler output. See the
> [roadmap checklist](#roadmap-checklist) for exactly what is left.

## Target hardware

- **Board:** Terasic DE1-SoC
- **FPGA:** Intel/Altera Cyclone V `5CSEMA5F31C6`
- **On-chip memory:** ~556 KB M10K SRAM (weights stored in block RAM)
- **Compute:** 87 DSP blocks (18×18 signed multipliers)
- **Clock:** 50 MHz (`CLOCK_50`, PLL configurable)

## Dataflow

```
┌──────────────────┐     ┌──────────────────┐     ┌───────────────────────┐
│  PyTorch Model   │ ──► │  fpGPT Compiler  │ ──► │  Weights (.mif/.hex)  │
│  (micro_gpt.pt)  │     │  quantize + emit │     │  + RTL wrappers/.vh   │
└──────────────────┘     └──────────────────┘     └───────────┬───────────┘
                                                             │
                                                             ▼
                                              ┌────────────────────────────┐
                                              │ Quartus synth/fit → .sof   │
                                              │ (NOT YET DONE)             │
                                              └────────────────────────────┘
```

On-chip, `fpga_top.v` connects UART to `generation_controller.v`, which wraps
`transformer_engine.v` and owns the unified weight ROM. The engine sequences
`embeddings → LN1 → attention → residual → LN2 → MLP → residual → final LN →
LM head`; the controller buffers a prompt, runs one forward pass per generated
character, selects the next token from the top-k logits, and feeds it back.

## What works today

- **Training:** a character-level Micro-GPT (`model/micro_gpt.py`) and a
  training script (`model/train.py`).
- **Compiler (`compile.py`):** loads a checkpoint, quantizes to INT8/INT16 with
  exact power-of-two scales, selects per-stage activation formats (defaults,
  calibration, or overrides), assigns a sequential memory layout, and exports:
  - per-tensor `.mif` / `.hex` weight files,
  - a byte-addressed little-endian `weights/weights_unified.hex`,
  - `rtl/gpt_params.vh` and a `rtl/weight_rom.v` ROM wrapper,
  - `fixed_point.json`, `rtl/fixed_params.vh`, configured `attention_blockN`
    and `fixed_*` wrappers, and GELU lookup tables.
- **RTL building blocks:** `attention.v` (full causal multi-head attention),
  `dense_layer.v`, `layer_norm.v`, `mac_unit.v`, `activation.v`, `rom_sync.v`,
  plus the numeric-contract operators `fixed_linear.v`,
  `fixed_layer_norm.v`, `fixed_residual.v`, `fixed_gelu.v`.
- **Full-pass engine:** `transformer_engine.v` runs a complete Micro-GPT
  forward pass in simulation and is checked bit-exactly against an independent
  Python integer reference (`tests/test_transformer_engine.py`).
- **Board-level generation:** `generation_controller.v` implements prompt
  buffering, character↔token mapping, LFSR-indexed top-k token selection
  (`TOP_K`, default 4), token feedback, a cycle counter, and the weight ROM;
  `fpga_top.v` wires it to UART. Verified end-to-end in simulation by
  `tests/test_generation_controller.py` and `tests/tb_fpga_top.v`.
- **FPGA project:** a synthesizable DE1-SoC target with a 50→150 MHz PLL
  (`pll_150.v`), a real SDC (`fpGPT.sdc`), a Quartus flow (`build.tcl`,
  `timing.tcl`), and a programming runbook (`PROGRAMMING.md`). The clock
  (`CLOCK_50`), reset keys (`KEY[3:0]`) and UART pins (GPIO_0[0:1]) are
  verified against the DE1-SoC pin table.
- **Throughput harness:** `tests/test_throughput.py` measures cycles/token in
  RTL and reports tok/s at 50 and 150 MHz (the default shape is gated behind
  `FPGPT_THROUGHPUT_DEFAULT=1`).
- **Attention verification:** `tests/test_attention.py` passes under Icarus
  Verilog.

## What is not done

- Quartus has **not** been run: no synthesis/fit, no timing closure, no area,
  and no `.sof`. `build.tcl`/`timing.tcl` and the SDC are in place, but the
  150 MHz target is unverified and may miss (the combinational dividers are the
  suspected critical path).
- The board has never been programmed or brought up; `PROGRAMMING.md` is the
  runbook, not a report.
- `compile.py` emits a uniform-width engine image
  (`build/weights/engine/weights_unified.hex`) and `build/rtl/board_params.vh`,
  which `fpga_top.v` consumes. Numerical agreement between that engine image and
  the fixed-point contract is still approximate (global shifts vs per-layer,
  8-bit vs accumulator-domain biases); see
  [docs/fixed_point.md](docs/fixed_point.md#integration-status).
- `hdl/gpt_controller.v` and `hdl/transformer_block.v` have been deleted.
- Sampling is LFSR-indexed top-k only (no temperature), and there is no output
  backpressure for the slower UART.
- The `UART_TXD`/`UART_RXD` pins are on GPIO_0 and need an external 3.3 V
  USB-TTL adapter; the onboard USB-UART belongs to the HPS.

## Quick start

### 0. Install dependencies

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# RTL simulation additionally needs Icarus Verilog (iverilog + vvp) on PATH.
```

### 1. Train a Micro-GPT

```bash
# Built-in sample text
python -m model.train --epochs 50

# Your own corpus
python -m model.train --data data/input.txt --epochs 100
```

Checkpoints are written to `checkpoints/micro_gpt.pt`.

### 2. Compile to Verilog and weight images

```bash
python compile.py --model checkpoints/micro_gpt.pt --out build/
# Optional: --precision 16, --calibration calibration.json, --formats overrides.json
```

### 3. Simulate the RTL

```bash
python3 -m unittest tests.test_attention -v
python3 -m unittest tests.test_transformer_engine -v
python3 -m unittest tests.test_generation_controller -v
python3 -m unittest tests.test_throughput -v
```

These run in `iverilog` and compare against independent Python integer models.
`test_throughput` also prints measured cycles/token and tok/s.

### 4. Build for the DE1-SoC

```bash
cd fpga
quartus_sh -t build.tcl     # syncs board_params/ROM, runs map/fit/asm/sta
quartus_sta -t timing.tcl   # Fmax and worst setup paths
```

Then program and test per [fpga/PROGRAMMING.md](fpga/PROGRAMMING.md). Quartus
has not been run in this repository yet, so treat the flow as unverified.

## Generated artifacts

`python compile.py` produces, under `--out`:

| Artifact | Purpose |
| --- | --- |
| `weights/*.mif`, `weights/*.hex` | Per-tensor weight/bias images at native width |
| `weights/weights_unified.hex` | Byte-addressed little-endian ROM image (all tensors) |
| `weights/engine/weights_unified.hex` | Uniform-width image consumed by the board engine |
| `weights/blockN_gelu.hex` | Per-block GELU lookup table |
| `rtl/gpt_params.vh` | Architecture constants and per-layer base addresses |
| `rtl/board_params.vh` | Board/engine localparams (`BOARD_*`, `ENGINE_*`) for `fpga_top.v` |
| `rtl/weight_rom.v` | ROM module using `$readmemh("weights/engine/weights_unified.hex")` |
| `rtl/fixed_params.vh` | Numeric constants (fractional bits, shifts, epsilon) |
| `rtl/attention_blockN.v`, `rtl/*_fixed.v` | Configured numeric wrappers |
| `fixed_point.json` | Numeric ABI manifest (formats, shifts, byte offsets) |
| `activation_formats.json` | Selected per-stage activation formats |

## Project structure

```
fpGPT/
├── compile.py                 # CLI: checkpoint → quantize → weights + RTL
├── requirements.txt
├── compiler/
│   ├── ir.py                  # Model IR and sequential memory layout
│   ├── quantizer.py           # Per-tensor power-of-two quantization
│   ├── fixed_contract.py      # Activation formats, biases, requant shifts
│   ├── calibration.py         # Per-stage scale selection from prompts
│   ├── mif_writer.py          # .mif/.hex plus unified little-endian ROM
│   └── fixed_export.py        # fixed_point.json, wrappers, GELU tables
├── model/
│   ├── micro_gpt.py           # Character-level GPT + tokenizer
│   ├── train.py               # Training script
│   ├── fixed_point.py         # Integer-reference forward pass + block emulator
│   └── validate_fixed.py      # Fixed vs PyTorch RMSE/error CLI
├── hdl/
│   ├── attention.v            # Causal multi-head attention, per-layer KV cache (verified)
│   ├── dense_layer.v          # Time-multiplexed linear engine
│   ├── layer_norm.v           # Sequential integer LayerNorm
│   ├── mac_unit.v             # DSP MAC primitive
│   ├── activation.v           # ReLU / piecewise GELU
│   ├── rom_sync.v             # M10K synchronous ROM
│   ├── fixed_linear.v         # Exact-scale linear operator
│   ├── fixed_layer_norm.v     # Exact-scale LayerNorm operator
│   ├── fixed_residual.v       # Scale-aligning residual add
│   ├── fixed_gelu.v           # Table-based GELU
│   └── transformer_engine.v   # Full forward-pass sequencer (verified)
├── fpga/
│   ├── fpga_top.v             # DE1-SoC top level (UART ↔ controller, PLL)
│   ├── generation_controller.v# Prompt buffer, top-k/argmax, feedback, ROM
│   ├── pll_150.v              # 50→150 MHz altera_pll wrapper
│   ├── board_params.vh        # Generated FPGPT_* board macro contract
│   ├── uart_rx.v, uart_tx.v
│   ├── fpGPT.qpf/.qsf/.sdc    # Quartus project, pins, timing constraints
│   ├── build.tcl, timing.tcl  # Non-interactive build + STA
│   └── PROGRAMMING.md         # Program/pof and UART bring-up runbook
├── tests/
│   ├── test_attention.py            # RTL vs integer reference (passes)
│   ├── test_transformer_engine.py   # Full-pass RTL test
│   ├── test_generation_controller.py# Board-level end-to-end RTL test
│   ├── test_throughput.py           # cycles/token + tok/s harness
│   ├── test_tokenizer.py            # Tokenizer ↔ hardware mapping contract
│   ├── test_fixed_point.py          # Numeric contract / integer reference
│   └── tb_fpga_top.v                # UART board smoke test (tiny model)
├── docs/
│   └── fixed_point.md         # Numeric contract and integration guide
└── hdl/attention.md           # Attention interface and numeric contract
```

## Default model configuration

| Parameter | Value | Notes |
| --- | --- | --- |
| Vocab size | 98 | 3 specials + full printable ASCII (32–126) |
| Context length | 64 | Tokens per inference window |
| d_model | 64 | Embedding dimension |
| Attention heads | 4 | Head dim = 16 |
| Transformer layers | 4 | Pre-norm decoder blocks |
| MLP hidden dim | 256 | 4× expansion |
| **Total params** | **~210K** | **~210 KB at INT8** |
| **M10K usage** | **~38%** | Within the 556 KB budget |

## Testing

- `make test` — **one command for the whole repo**: Python unit tests, Icarus
  elaboration of every HDL module, the Verilator width/latch lint gate, and the
  board smoke test. Requires `iverilog`, `vvp` and `verilator` on `PATH`
  (Python tests still run without them). `make test-python`, `make test-hdl`
  and `make test-board` run the pieces individually.
- `python3 -m unittest tests.test_attention -v` — attention RTL vs an
  independent integer reference. Requires `iverilog`/`vvp`; skips otherwise.
- `python3 -m unittest tests.test_transformer_engine -v` — full forward pass in
  RTL vs a Python integer model.
- `python3 -m unittest tests.test_generation_controller -v` — UART-style prompt
  through `generation_controller` to a generated byte, vs the integer
  reference. Exercises the whole board datapath in simulation.
- `python3 -m unittest tests.test_throughput -v` — runs a generation burst in
  RTL, counts cycles, and prints tok/s at 50/150 MHz. Set
  `FPGPT_THROUGHPUT_DEFAULT=1` to measure the default model shape.
- `iverilog ... tests/tb_fpga_top.v` — UART board smoke test with a tiny
  zeroed model; the run command is in the file header.
- `python3 -m unittest discover -s tests -v` — collects everything, including
  `test_fixed_point.py` (integer-reference vs PyTorch and RTL checks). It must
  be run through `discover` because it imports `test_attention` by module name;
  RTL tests skip when `iverilog`/`vvp` are not on `PATH`.
- `pytest tests/` — `pytest` is pinned in `requirements.txt`; the suite also
  runs under `unittest`.

## Documentation

- [docs/fixed_point.md](docs/fixed_point.md) — fixed-point numeric contract,
  calibration, generated hardware, and known integration limits.
- [hdl/attention.md](hdl/attention.md) — attention RTL interface and numeric
  contract.

## Roadmap checklist

`[x]` = implemented and verified in simulation; `[ ]` = still to do.

### Numerics and compiler
- [x] Exact power-of-two weight scales (INT8/INT16)
- [x] 64-bit accumulator-domain biases and per-stage requant shifts
- [x] Per-stage activation formats: defaults, calibration, and overrides
- [x] Per-tensor `.mif`/`.hex` export plus unified little-endian ROM image
- [x] GELU lookup-table export
- [x] Configured `attention_blockN` / `fixed_*` wrappers and `fixed_point.json`
- [x] Restore the `model/fixed_point.py` integer-reference API
      (`FixedMicroGPT`, `attention`, `layer_norm`, `residual`, `gelu_table`,
      `shift`, `clip`) required by `model/validate_fixed.py` and
      `tests/test_fixed_point.py`
- [ ] Reconcile `transformer_engine.v` / `generation_controller.v`
      (8-bit sign-extended biases, 8-bit LayerNorm params) with the compiler's
      mixed-width ROM (64-bit biases, 32-bit Q14 gamma / 32-bit beta); a
      uniform-width engine image is emitted and consumed today, but numerical
      agreement, per-layer shifts and accumulator-domain biases still need
      validation/wiring
- [x] Emit board parameters (`ROM_DEPTH`, `W_ADDR_WIDTH`, shifts) and the
      uniform-width engine image from `compile.py` (`build/rtl/board_params.vh`,
      consumed by `fpga_top.v`)
- [ ] Emit a configured `generation_controller` wrapper and wire it into
      `fpga_top.v` instead of hand-maintained defaults

### RTL and system integration
- [x] `attention.v`: full causal multi-head attention
- [x] `dense_layer.v`, `layer_norm.v`, `mac_unit.v`, `activation.v`,
      `rom_sync.v`
- [x] `fixed_linear.v`, `fixed_layer_norm.v`, `fixed_residual.v`,
      `fixed_gelu.v`
- [x] `transformer_engine.v`: complete forward-pass sequencer
- [x] `generation_controller.v`: prompt buffer, byte↔token mapping, top-k
      selection, token feedback, weight ROM
- [x] `fpga_top.v` wired to `generation_controller` over UART
- [x] Delete the unused `transformer_block.v` and `gpt_controller.v` stubs
- [ ] Add output buffering/backpressure (UART TX is slower than the engine)
- [x] Per-layer KV cache / incremental decoding in `attention.v` +
      `transformer_engine.v` (decode only processes the new token)
- [x] LFSR-indexed top-k sampling in `generation_controller.v` (verified)
- [ ] Add temperature control and finish the sampling policy
- [ ] Define and document a stable UART/host protocol (prompt framing,
      generation length, status)

### FPGA and board
- [x] `fpga_top.v`, `generation_controller.v`, `uart_rx.v`, `uart_tx.v`,
      Quartus project, SDC, LED debug
- [x] Quartus project includes the real engine path (`transformer_engine.v`,
      `attention.v`, `layer_norm.v`, `dense_layer.v`, `mac_unit.v`,
      `activation.v`, `rom_sync.v`)
- [x] 50→150 MHz PLL (`pll_150.v`) and a real SDC (`fpGPT.sdc`) constraining
      `CLOCK_50` and the generated `clk_sys`
- [x] Non-interactive Quartus flow (`build.tcl`, `timing.tcl`) and a
      program/pof + UART runbook (`PROGRAMMING.md`)
- [x] Resolve the ROM `$readmemh` path (`SEARCH_PATH` + `MIF_FILE` +
      `build.tcl` copying the compiled engine image)
- [x] Verify clock/reset/UART pin assignments against the DE1-SoC pin table
      (`CLOCK_50`, `KEY[3:0]`, GPIO_0[0:1])
- [ ] Run Quartus synthesis/fit; record ALM, DSP, and M10K usage
- [ ] Achieve timing closure and record Fmax (150 MHz is unverified)
- [ ] On-board bring-up and end-to-end generation test

### Verification and tooling
- [x] Attention RTL vs integer-reference tests
- [x] Full forward-pass RTL test (`test_transformer_engine.py`)
- [x] Board-level end-to-end RTL test (`test_generation_controller.py`)
- [x] Board UART smoke test updated for the engine (`tb_fpga_top.v`)
- [x] Throughput harness (`test_throughput.py`)
- [x] Restore `tests/test_fixed_point.py` collection (integer-reference vs
      PyTorch / RTL)
- [x] Add `pytest` to `requirements.txt`
- [x] Elaborate the PLL/board top in CI without Quartus
      (`test_board_top_elaborates_with_pll`, via the `altera_pll` stub)
- [x] Delete the orphaned `tb_transformer_block.v`
- [ ] Add CI that installs `torch` and `iverilog` and runs the full suite
- [ ] Add an end-to-end test from checkpoint → compiled ROM → board RTL

### Model and documentation
- [x] `MicroGPT` model and training script
- [ ] Provide a trained checkpoint and a generation-quality baseline
- [ ] Keep this README and the docs in sync as work lands

## License

MIT
