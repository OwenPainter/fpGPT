# fpGPT TODO — performance, correctness, and board bring-up

Target: **~500 generated tokens/second** on the DE1-SoC, with correct
fixed-point inference. Baseline is unmeasured; add the cycle counter first.

Legend: impact on tok/s, effort, and the files involved.

## P0 — correctness blockers (must land before a flashed model can be trusted)

- [ ] **Reconcile ROM widths between compiler and engine.** `transformer_engine.v`
      assumes every tensor is `DATA_WIDTH` (8-bit), but `compiler/quantizer.py`
      now runs `fixed_contract.configure()`, which stores 64-bit
      accumulator-domain biases and 32-bit Q14 gamma / 32-bit beta in the
      unified byte ROM (`compiler/mif_writer.py`). The engine therefore reads
      the generated image with the wrong widths. Converge on one contract
      (`docs/fixed_point.md`) and either:
      - make the engine consume the mixed-width image, or
      - generate an all-`DATA_WIDTH` image for the engine.
      Files: `hdl/transformer_engine.v`, `compiler/mif_writer.py`,
      `compiler/fixed_contract.py`, `hdl/attention.v`, `hdl/layer_norm.v`.
- [ ] **Restore the integer-reference API in `model/fixed_point.py`**
      (`FixedMicroGPT`, `attention`, `layer_norm`, `residual`, `gelu_table`,
      `shift`, `clip`) so `model/validate_fixed.py` and
      `tests/test_fixed_point.py` import. Backup of the integer-only version:
      `/tmp/fixed_point.local.py` (merge with the existing
      `FixedPointTransformerBlock`).
- [ ] **Make `compile.py` emit the board parameters** (`ROM_DEPTH`,
      `W_ADDR_WIDTH`, per-stage shifts, model dims) and a configured
      `generation_controller`/engine wrapper, so `fpga_top.v` defaults cannot
      drift from the compiled model by hand.
- [x] **Fix the ROM init path for Quartus.** `rom_sync` uses
      `ROM_MEM_FILE="weights_unified.hex"`; `build.tcl` copies the compiled
      engine image into `fpga/`, the QSF adds `SEARCH_PATH
      ../build/weights/engine` and registers it via `MIF_FILE`, and
      `VERILOG_INCLUDE_FILE` handles `board_params.vh`. M10K inference is not
      yet confirmed by a fit.
- [x] **Add a cycle counter + throughput testbench** so every change is
      quantified in tok/s (`generation_controller.v` exposes `gen_cycles`;
      `tests/test_throughput.py` reports tok/s). Measured baseline for the
      default shape (d_model=64, 4 layers): **963,686 cycles/token → 51.9 tok/s
      @ 50 MHz, 155.7 tok/s @ 150 MHz**. The 500 tok/s target still needs the
      P1/P2 work below.

## P1 — algorithmic speedups (largest tok/s gains)

- [ ] **KV cache (~30–64x at context 64).** Split the engine into
      **prefill** (prompt, full sequence) and **decode** (one new token/step);
      cache K/V per layer in M10K (~32 KB for the default config). This removes
      the O(T²) recompute and the per-token full-sequence pass.
      Files: `hdl/transformer_engine.v`, `hdl/attention.v`,
      `fpga/generation_controller.v`.
- [ ] **Weight residency.** Stop reloading all `4*D_MODEL^2` attention weights
      from ROM on every layer/token; keep the active layer's weights resident
      (per-layer cores or a cached buffer), swapping only on layer change.
- [ ] **Parameterized parallel MAC array (2–64x).** Make `NUM_PES` real in
      `hdl/dense_layer.v`: output-stationary PEs using the 87 DSP blocks.
      - fc1 (OUT=256, IN=64): 64 PEs → one dot product/cycle, ~256 cycles.
      - fc2 / projections (OUT=64, IN=64): 64 PEs → ~64 cycles.
      - Shared array across stages is acceptable if the schedule serializes.

## P2 — clock, pipeline, and memory

- [ ] **Raise the clock toward ~150 MHz** via the PLL (currently 50 MHz).
- [ ] **Remove combinational dividers** from `hdl/attention.v` (context
      normalize) and `hdl/layer_norm.v` (mean/variance/inv); use reciprocal LUTs
      or iterative pipelined dividers. These are the likely Fmax limiter.
- [ ] **Pipeline the MAC and ROM read** (deeper than the current 2 stages) and
      register the softmax/exp path.
- [ ] **Bank the unified ROM across M10K blocks** and use wide/dual-port reads
      so a 64-wide PE array gets 64 weight bytes/cycle.
- [ ] **Overlap stages** (e.g., LN of the next token with the MLP of the
      current one) once the decode path is a clean pipeline.

## P3 — quality ("better")

- [ ] **Calibration-driven per-stage formats** (`compiler/calibration.py`) to
      reduce clipping; validate with `model/validate_fixed.py` RMSE.
- [ ] **INT4 option** to fit a larger model (more layers / wider `d_model`) in
      the same 556 KB M10K budget.
- [ ] **Better numerics**: exact-erf GELU LUT (already exported), a proper
      reciprocal/exp table for softmax, per-tensor scale tracking.
- [ ] **Sampling**: temperature and top-k instead of greedy argmax
      (quality, not speed); document the host protocol.
- [ ] **Larger / better-trained model**: more data, longer context, a real
      tokenizer (currently 64-char ASCII).

## P4 — board bring-up

- [ ] **Timing/area closure in Quartus** (no `fit`/STA has been run; no `.sof`).
      `fpga/build.tcl`, `fpga/timing.tcl`, `pll_150.v` and `fpGPT.sdc` are in
      place, but 150 MHz is unverified and likely misses.
- [x] **Verify pin assignments** in `fpga/fpGPT.qsf` — UART is on GPIO_0[0]
      (`PIN_AC18`) / GPIO_0[1] (`PIN_Y17`), confirmed against the DE1-SoC pin
      table.
- [x] **HPS or GPIO UART decision** — external 3.3 V USB-TTL adapter on
      GPIO_0, documented in `fpga/PROGRAMMING.md` (the onboard USB-UART is HPS).
- [ ] **On-board smoke test**: program the `.sof` via USB-Blaster, open a
      115200 8N1 terminal, send a prompt + CR/LF, confirm streamed characters.
      `tests/tb_fpga_top.v` now exercises the engine path in simulation.
- [ ] **Persistent config**: convert `.sof` to `.pof`/`.jic` for flash boot
      (steps documented in `fpga/PROGRAMMING.md`).

## Critical path to a working 500 tok/s demo

1. Fix the ROM width mismatch (P0) → correct inference in simulation.
2. Add cycle counter + throughput test (P0) → measurable baseline.
3. KV cache prefill/decode split (P1) → the dominant speedup.
4. PE array + 150 MHz + divider removal (P1/P2) → clear 500 tok/s.
5. Board bring-up + timing closure (P4) → run it on hardware.
