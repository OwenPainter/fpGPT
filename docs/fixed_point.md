# Fixed-point numerical contract

This document describes the exact-scale numeric contract emitted by the
compiler (`compiler/fixed_contract.py`, `compiler/fixed_export.py`) and
implemented by the `fixed_*` RTL operators. It is **not** the contract used by
`transformer_engine.v`; that engine currently consumes simple integer
weights/biases through `dense_layer.v`/`layer_norm.v` and is not yet wired to
these exports. See [Integration status](#integration-status).

A stored integer `q` with fractional bits `F` represents `q * 2**(-F)`.
Weight scales are exact powers of two, not arbitrary scales approximated later
by a shift. Model precision selects INT8/INT16 activation and weight width;
biases and LayerNorm parameters use wider storage.

## Compile and validate

Install `requirements.txt` and Icarus Verilog for simulation tests. Calibration
JSON contains a list of representative token-ID sequences, for example
`[[3, 4, 5], [6, 7, 8]]`. Use real prompts within the checkpoint's vocabulary
and context length.

```bash
python compile.py --model checkpoints/micro_gpt.pt --out build/ \
  --precision 8 --calibration calibration.json

python -m model.validate_fixed --model checkpoints/micro_gpt.pt \
  --precision 8 --formats build/activation_formats.json \
  --tokens 3,4,5 --max-rmse 0.05

python -m unittest discover -s tests -v
```

> **Resolved:** `model/fixed_point.py` now defines the integer-reference API
> (`FixedMicroGPT`, `attention`, `layer_norm`, `residual`, `gelu_table`,
> `shift`, `clip`) alongside the block-level emulator, so the validation
> command and `tests/test_fixed_point.py` run.

Calibration chooses per-stage fractional bits from the maximum observed
absolute activation with 10% headroom. Embedding ranges include the entire
tables. `--formats overrides.json` overrides selected stages: the JSON maps
stage names to fractional-bit integers in 0..15. Generated
`activation_formats.json` contains the full selection for reproducibility.
Without calibration, conservative fixed defaults are used. Calibration does
not guarantee that unseen sequences will avoid clipping.

The validation command reports RMSE, maximum absolute error, and boundary-value
counts for each recorded stage. Boundary counts are diagnostic, not exact
saturation counts. Optional `--max-rmse` enforces a caller-selected final-logit
threshold. Validate on held-out sequences and assess generation quality
separately; there is no universal acceptable error threshold.

## Arithmetic

| Quantity / operation | Contract |
| --- | --- |
| Weight | Signed INT8/16; largest fitting binary fractional count, capped at 24 |
| Activation | Signed INT8/16; independently selected per named stage |
| Parameter conversion | Nearest, ties to even; reject non-finite/unrepresentable explicit formats |
| MAC | Signed 64-bit sum of full-width products |
| Bias | Signed INT64, `F_bias = F_input + F_weight`; add before rescaling |
| Linear output | Shift by `F_input + F_weight - F_output`, then saturate |
| Right shift | Arithmetic, toward negative infinity |
| Negative shift | Left shift; compiler checks conservative overflow bounds |
| Residual/embedding sum | Align to finest operand/output scale, add in 64 bits, rescale and saturate once |
| LayerNorm mean | Inputs retain 8 extra fractional bits; signed division by dimension toward zero |
| LayerNorm variance | Floored population mean of squared centered values |
| LayerNorm epsilon | `max(1, round(eps * 2**(2*(F_input+8))))`, in variance units |
| LayerNorm root | Integer floor square root of variance plus epsilon |
| Normalized vector | Signed Q14, division toward zero |
| LayerNorm affine | Gamma INT32/Q14, beta INT32 at output scale; shift product then add beta and saturate |
| Attention score | Quarter-unit integer including Q/K scales and inverse square root of head dimension |
| Softmax exponential | `round(32768 * exp(-delta/4))`, zero for delta > 32, max-subtracted |
| Attention context | `sum(exp * V) / sum(exp)`, division toward zero; retains V scale |
| GELU | Offline exact-erf table rounded to output scale; runtime integer lookup |

Attention uses `SCORE_MULT=round(2**20/sqrt(head_dim))` and
`SCORE_SHIFT=20+F_q+F_k-2`. Probabilities are not rounded individually. At least
one exponential is 32768, guaranteeing a nonzero denominator.

The quarter-unit score grid and finite exponential tail remain approximations
in INT16. Higher activation width does not eliminate their error. LayerNorm's
integer mean/root and floor shifts also introduce error. The integer reference
reproduces these operations rather than substituting floating-point functions.

## Generated hardware and storage

`fixed_point.json` records formats, shifts, bias scales, LayerNorm epsilon,
attention settings, and byte offsets. `rtl/fixed_params.vh` exposes constants.
Generated wrappers instantiate:

- `attention_blockN`: `attention.v` with block-specific shifts and score settings.
- Linear `*_fixed` wrappers: `fixed_linear.v` with the projection's output shift.
- LayerNorm `*_fixed` wrappers: `fixed_layer_norm.v` with stage epsilon/output format.
- `embedding_add`, `blockN_residual1/2`: `fixed_residual.v` with operand alignment.
- `blockN_gelu`: `fixed_gelu.v` using `weights/blockN_gelu.hex`.

Linear and LayerNorm cores process one vector per start, emit ordered `y_valid`
outputs without backpressure, and pulse `done`. Load all parameters/inputs
while idle and deassert load before start. Load zero biases where appropriate.
Reset aborts work and preserves memories. GELU has one cycle of latency;
residual addition is combinational. Compile wrappers together with `hdl/`
modules; run simulations from the build directory for relative ROM paths.

Attention images group Q, K, V and output weights in that order. Matching
`blockN_attention_biases.hex` files use 64-bit product-scale biases. Other
tensor `.hex`/`.mif` files retain each tensor's native width. The unified ROM
is **byte addressed, little endian**, including INT16 mode: a loader must
assemble 2-byte weights, 4-byte LayerNorm parameters and 8-byte biases using
the manifest. The compiler writes `weights/weights_unified.hex` in every export
mode.

`dense_layer.v`, `layer_norm.v` and `activation.v` are not dependencies of the
`fixed_*` wrappers; use the configured operators for this contract. They are,
however, used by `transformer_engine.v`, which implements a different integer
scheme.

GELU tables add 256 bytes per INT8 stage or 128 KiB per INT16 stage beyond the
weight budget; the compiler reports this separately. Exports do not currently
promise physical table sharing. LayerNorm/attention use combinational division
and LayerNorm has an unrolled integer square root. These are synthesizable
reference datapaths; Quartus area, memory mapping and timing have not been
measured.

## Integration status

- The `fixed_*` operators are generated and elaborated in simulation, but no
  controller sequences them into a full model.
- `hdl/transformer_engine.v` **does** sequence a full forward pass and now
  consumes a compiler-generated **uniform-width** image
  (`build/weights/engine/weights_unified.hex`). Its numeric contract is still
  approximate: it takes one global requant shift per projection type, while the
  fixed contract assigns per-layer shifts, and it adds `DATA_WIDTH` biases
  directly rather than in the accumulator domain.
- `model/fixed_point.py` defines both the integer-reference API the validation
  tooling imports and the older block-level emulator.
- `compile.py` also emits a **uniform-width** engine image
  (`build/weights/engine/weights_unified.hex`) and `build/rtl/board_params.vh`
  for the ROM-based engine, separate from the mixed-width fixed-contract image.
- Board integration now exists in simulation: `fpga/generation_controller.v`
  wraps `transformer_engine.v`, owns the ROM (`rom_sync`), buffers a prompt,
  selects a token from the top-k logits (`TOP_K`), and feeds it back;
  `fpga/fpga_top.v` wires it to UART. It inherits the width mismatch above, has
  no KV cache that skips recomputation, and has no documented host protocol.
  `gpt_controller.v` and `transformer_block.v` have been deleted.

## Verification

`tests/test_fixed_point.py` requires exact RTL/integer agreement for attention,
linear layers, LayerNorm, residuals and GELU, and PyTorch comparisons of
attention, LayerNorm and full-model logits at both precisions. The RTL portions
skip when `iverilog`/`vvp` are not on `PATH`; the Python numeric checks run
regardless.

What is verified today:

- `tests/test_attention.py` passes: attention RTL vs an independent Python
  integer reference, covering multiple heads, non-power-of-two head dimensions,
  signed values, distinct projection shifts, biases, 8/16-bit data, saturation,
  uniform attention, causal isolation, full/partial lengths, default model
  dimensions, reset during execution, invalid lengths, ignored busy-time
  writes, and repeated inference.
- `tests/test_transformer_engine.py` runs the full forward pass in RTL and
  compares the last token's vocabulary projection against an independent Python
  integer model. It uses its own 8-bit ROM layout, not compiler output.
- `tests/test_generation_controller.py` drives `generation_controller.v` the way
  UART would, with a tiny model ROM, and checks that the emitted byte matches
  the argmax of the integer reference's final-position logits (the test pins
  `TOP_K=1`, i.e. greedy).
- `tests/tb_fpga_top.v` runs the same UART path through `fpga_top.v` with a tiny
  zeroed model and checks a character is generated.
- `tests/test_throughput.py` measures cycles/token in RTL and reports tok/s at
  the 50 MHz and 150 MHz targets.

Tolerances in the fixed-point tests apply to deterministic fixtures, not
arbitrary trained checkpoints.
