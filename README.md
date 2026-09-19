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
  (`TOP_K`, default 4), token feedback, and the weight ROM; `fpga_top.v` wires
  it to UART. Verified end-to-end in simulation by
  `tests/test_generation_controller.py`.
- **FPGA scaffolding:** UART RX/TX, LED debug, and a Quartus project
  (`fpga/fpGPT.qpf`, `.qsf`, `.sdc`).
- **Attention verification:** `tests/test_attention.py` passes under Icarus
  Verilog.

## What is not done

- The board ROM is **not** compatible with the compiler's current unified ROM:
  `generation_controller`/`transformer_engine` expect 8-bit weights, 8-bit
  sign-extended biases, and 8-bit LayerNorm parameters, while the compiler
  exports 64-bit attention biases and 32-bit LayerNorm gamma/beta. See
  [docs/fixed_point.md](docs/fixed_point.md#integration-status).
- `compile.py` emits `build/rtl/board_params.vh` (ROM depth/address width, model
  dimensions, per-layer requant shifts) and a uniform-width engine image at
  `build/weights/engine/weights_unified.hex`, so `fpga_top.v`'s defaults no
  longer have to be synced by hand.
- `model/fixed_point.py` defines the integer-reference API imported by
  `model/validate_fixed.py` and `tests/test_fixed_point.py`; those run.
- The top-k selection added to `generation_controller.v` is unverified: the RTL
  test suite could not be run because `iverilog`/`vvp` are not installed in this
  environment.
- There is no KV cache that actually skips recomputation (each token still runs
  the full sequence), no temperature sampling, and no synthesis, timing, or
  board bring-up.
- `tests/tb_fpga_top.v` still tests the old echo stub and is stale;
  `tests/tb_transformer_block.v` is an orphaned testbench.

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
```

These run in `iverilog` and compare against independent Python integer models.

### 4. Synthesize with Quartus (not yet verified)

Open `fpga/fpGPT.qpf` in Quartus Prime Lite, compile, and program the DE1-SoC.
`generation_controller` loads `weights/weights_unified.hex` (relative to the
run directory), so the ROM image and a correctly parameterized top level must
be provided first. This has not been brought up on hardware.

## Generated artifacts

`python compile.py` produces, under `--out`:

| Artifact | Purpose |
| --- | --- |
| `weights/*.mif`, `weights/*.hex` | Per-tensor weight/bias images at native width |
| `weights/weights_unified.hex` | Byte-addressed little-endian ROM image (all tensors) |
| `weights/blockN_gelu.hex` | Per-block GELU lookup table |
| `rtl/gpt_params.vh` | Architecture constants and per-layer base addresses |
| `rtl/weight_rom.v` | ROM module using `$readmemh("weights/weights_unified.hex")` |
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
│   ├── fixed_point.py         # Integer-reference operators
│   └── validate_fixed.py      # Fixed vs PyTorch CLI
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
│   ├── fpga_top.v             # DE1-SoC top level (UART ↔ controller)
│   ├── generation_controller.v# Prompt buffer, top-k selection, token feedback, ROM
│   ├── uart_rx.v, uart_tx.v
│   └── fpGPT.qpf, fpGPT.qsf, fpGPT.sdc
├── tests/
│   ├── test_attention.py            # RTL vs integer reference (passes)
│   ├── test_transformer_engine.py   # Full-pass RTL test
│   ├── test_generation_controller.py# Board-level end-to-end RTL test
│   ├── test_fixed_point.py          # Integer-reference vs PyTorch / RTL
│   └── tb_fpga_top.v, tb_transformer_block.v  # Stale/orphaned testbenches
├── docs/
│   └── fixed_point.md         # Numeric contract and integration guide
└── hdl/attention.md           # Attention interface and numeric contract
```

## Default model configuration

| Parameter | Value | Notes |
| --- | --- | --- |
| Vocab size | 64 | Printable ASCII subset |
| Context length | 64 | Tokens per inference window |
| d_model | 64 | Embedding dimension |
| Attention heads | 4 | Head dim = 16 |
| Transformer layers | 4 | Pre-norm decoder blocks |
| MLP hidden dim | 256 | 4× expansion |
| **Total params** | **~210K** | **~210 KB at INT8** |
| **M10K usage** | **~38%** | Within the 556 KB budget |

## Testing

- `python3 -m unittest tests.test_attention -v` — attention RTL vs an
  independent integer reference. Requires `iverilog`/`vvp`; skips otherwise.
- `python3 -m unittest tests.test_transformer_engine -v` — full forward pass in
  RTL vs a Python integer model.
- `python3 -m unittest tests.test_generation_controller -v` — UART-style prompt
  through `generation_controller` to a generated byte, vs the integer
  reference. Exercises the whole board datapath in simulation.
- `python3 -m unittest discover -s tests -v` — collects everything, including
  `test_fixed_point.py` (its integer-reference vs PyTorch and RTL checks).
  `test_fixed_point.py` must be run through `discover` because it imports
  `test_attention` by module name. RTL tests skip when `iverilog`/`vvp` are not
  on `PATH`.

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
      uniform-width engine image is emitted today, but per-layer shifts and the
      accumulator-domain biases still need wiring
- [x] Emit board parameters (`ROM_DEPTH`, `W_ADDR_WIDTH`, shifts) and the
      uniform-width engine image from `compile.py`
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
- [ ] Add a KV cache / incremental decoding (the engine still recomputes the
      whole sequence for every token)
- [ ] Verify and finish sampling: top-k selection is present but unverified,
      and there is no temperature control
- [ ] Define and document a stable UART/host protocol (prompt framing,
      generation length, status)

### FPGA and board
- [x] `fpga_top.v`, `generation_controller.v`, `uart_rx.v`, `uart_tx.v`,
      Quartus project, SDC, LED debug
- [x] Quartus project includes the real engine path (`transformer_engine.v`,
      `attention.v`, `layer_norm.v`, `dense_layer.v`, `mac_unit.v`,
      `activation.v`, `rom_sync.v`)
- [ ] Place the generated engine image (`weights/engine/weights_unified.hex`)
      where Quartus can find it and remove the relative-path fragility
- [ ] Run Quartus synthesis/fit; record ALM, DSP, and M10K usage
- [ ] Achieve timing closure and record Fmax
- [ ] Verify UART pin assignments against actual DE1-SoC wiring
- [ ] On-board bring-up and end-to-end generation test

### Verification and tooling
- [x] Attention RTL vs integer-reference tests
- [x] Full forward-pass RTL test (`test_transformer_engine.py`)
- [x] Board-level end-to-end RTL test (`test_generation_controller.py`)
- [x] Restore `tests/test_fixed_point.py` collection (integer-reference vs
      PyTorch / RTL)
- [ ] Replace or delete the stale `tb_fpga_top.v` / orphaned
      `tb_transformer_block.v`
- [ ] Install `iverilog`/`vvp` and actually run the RTL tests
- [ ] Add CI that installs `torch` and `iverilog` and runs the full suite
- [ ] Add an end-to-end test from checkpoint → compiled ROM → board RTL

### Model and documentation
- [x] `MicroGPT` model and training script
- [ ] Provide a trained checkpoint and a generation-quality baseline
- [ ] Keep this README and the docs in sync as work lands

## License

MIT
