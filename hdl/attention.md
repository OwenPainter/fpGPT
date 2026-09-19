# Causal attention RTL

`attention.v` implements the attention sublayer of `CausalSelfAttention`:
Q/K/V affine projections, scaled dot-product attention per head, an implicit
causal mask, approximate softmax, head concatenation, and the output affine
projection. Dropout is omitted for inference. LayerNorm and residual addition
belong outside this module.

## Host/controller interface

All inputs are sampled on the rising clock edge. Reset is active low and aborts
the current run; it does not erase the input or parameter memories.

1. While `busy=0`, load all input elements with `x_load`, all four weight
   matrices with `w_load`, and all four bias vectors with `b_load`. Each port
   accepts one element per clock; ports can operate simultaneously. Load zeros
   for unused biases. Memories have no implicit initialization.
2. Deassert load strobes, set `seq_len` in `1..MAX_SEQ_LEN`, and pulse `start`.
3. Capture each `y_data` when `y_valid=1`, at the flattened address `y_addr`.
   Outputs are emitted in token-major order. There is no backpressure; the
   receiving buffer must accept every valid output.
4. `done` pulses for one cycle after the final output, and `busy` returns low.
   Another sequence can then be loaded and started; weights may be reused.

Start and load strobes are ignored while busy. Out-of-range load addresses are
ignored. An invalid sequence length returns `done=1,error=1` without starting
work; `error` stays set until reset or the next accepted start. Only valid
sequence elements need loading. Pulsing reset requires waiting until a clock
edge after reset release before starting a run.

The address layouts are:

| Port | Address | Contents |
| --- | --- | --- |
| Input/output | `token * D_MODEL + channel` | Signed DATA_WIDTH activation |
| Weights | `projection * D_MODEL² + output_channel * D_MODEL + input_channel` | Signed DATA_WIDTH weight |
| Biases | `projection * D_MODEL + output_channel` | Signed 64-bit accumulator-domain bias |

Projection indices are Q=0, K=1, V=2, output=3. Heads are consecutive channel
slices of `D_MODEL / NUM_HEADS`, matching the Python model's reshape.

## Numeric contract

Supported data widths are 2..16 bits; model size must be divisible by head
count. Intermediate accumulators are signed 64-bit. Select dimensions/scales
so dot products, bias sums, score multiplication, and value sums fit 64 bits.
The module saturates projections and outputs to the signed DATA_WIDTH range.

Each projection computes:

```
sat((sum(input_integer * weight_integer) + bias_integer) >>> PROJECTION_SHIFT)
```

The arithmetic shift rounds negative numbers toward negative infinity. Biases
must use the product scale, before the shift. The compiler's
`blockN_attention_biases.hex` images already contain 64-bit product-scale
biases; independently quantized INT8 biases cannot be loaded directly without
rescaling. Q, K, V and output shifts are independently configurable.

Attention scores use quarter-unit integers:

```
score = (sum(q_integer * k_integer) * SCORE_MULT) >>> SCORE_SHIFT
```

If Q and K represent real values `q_integer * s_q` and `k_integer * s_k`, choose
`SCORE_MULT / 2**SCORE_SHIFT` approximately equal to
`4 * s_q * s_k / sqrt(HEAD_DIM)`. Defaults (`64/256` for HEAD_DIM=16) assume
`s_q=s_k=1/2`. Changing head size or projection scales requires changing these
parameters; the defaults are not inferred from a checkpoint.

Only keys `0..query_token` are visited, so future tokens cannot contribute.
For each score row, the maximum is subtracted, and a constant lookup table
computes `round(32768 * exp(-difference/4))`. Differences greater than 32
(eight real score units) become zero. The maximum always contributes 32768,
so the denominator is nonzero. For each value channel, the module computes:

```
context_integer = sum(exp_integer * v_integer) / sum(exp_integer)
```

Signed division truncates toward zero. The context retains the V scale.
Score quantization, the exponential table, clipping, and projection rounding
make this an approximate fixed-point implementation, not a bit-exact PyTorch
floating-point forward pass.

## Resource and integration status

The FSM executes one projection/score/value product at a time with separate
registered memory-read cycles. It buffers a whole sequence's inputs, Q/K/V and
context, plus four projection matrices and one score row. It does not use a
KV cache. Counters/address arithmetic and the 64-bit combinational divider
favor implementation simplicity; division can be expensive and may limit
clock frequency. No Cyclone V resource usage or timing closure is claimed.
Quartus synthesis and fitting are still required to establish these figures.

`compile.py` generates configured `attention_blockN` wrappers, projection
weight/bias loading images, and a numeric manifest. See
[the fixed-point contract](../docs/fixed_point.md). A surrounding transformer
controller must still load the proper block's weights and sequence activations
and capture the output; the wrapper does not autonomously read the unified ROM.

`hdl/transformer_engine.v` currently drives `attention.v` directly with raw
`DATA_WIDTH` weights and sign-extended 8-bit biases, not the `attention_blockN`
wrapper or the 64-bit product-scale images. Reconciling those two numeric paths
is pending; see [the integration status](../docs/fixed_point.md#integration-status).

## Verification

Run `python3 -m unittest discover -s tests -v` with `iverilog` and `vvp` on PATH.
Tests load the module through its public ports and compare every output to an
independent Python integer reference. Cases cover multiple heads, non-power-of-two
head dimensions, signed values, distinct projection shifts, biases, 8/16-bit
data, saturation, uniform attention, causal isolation, full/partial lengths,
default model dimensions, reset during execution, invalid lengths, ignored
busy-time writes/start, and repeated inference with retained parameters.
