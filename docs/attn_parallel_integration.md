# Parallel attention integration — exact patch plan

**Status: NOT APPLIED.** This is a reviewable plan for replacing the serial
projection loop in `hdl/attention.v` with `hdl/attn_proj.v`, plus the four
resident attention weight caches it needs in `hdl/transformer_engine.v`.

Prereqs already landed (verified standalone):
- `hdl/attn_proj.v` — 4 time-multiplexed `dense_layer` PE arrays, one packed
  weight bus per projection, per-projection shift/bias.
- `hdl/weight_cache.v` — packed/banked resident cache. Read rules:
  weights `raddr = j*IN_FEATURES + i` with `gather=1`; biases `raddr = j` with
  `gather=0`. The read and `gather` are registered, so data arrives two cycles
  after training the combined `(raddr, gather)` pair.
- `tests/test_attn_proj.py` — bit-exact vs reference, `NUM_PES` 1/2/4/8 and
  `cache_len > 0` multi-token ranges. 10 tests green in <3 s.

Owner: whoever is out of `attention.v` / `transformer_engine.v` when this lands.
Do **not** apply while a Quartus build is mid-flight.

---

## 0. Interface contract recap (`hdl/attn_proj.v`)

| signal | width | meaning |
|---|---|---|
| `start` | 1 | pulse high for one cycle to begin one projection |
| `projection` | 2 | `0=Q 1=K 2=V 3=OUT`, held stable for the whole run |
| `length` | `LEN_WIDTH=clog2(MAX_SEQ_LEN+1)` | exclusive token end |
| `cache_len` | same | first token to process; run is `[cache_len, length)` |
| `w_addr_{q,k,v,o}` | `W_ADDR_WIDTH` | stage-relative packed read address |
| `w_gather_{q,k,v,o}` | 1 | 1=weights, 0=bias |
| `w_data_{q,k,v,o}` | `NUM_PES*DATA_WIDTH` | packed packed bytes |
| `x_addr` | `X_AW=clog2(D_MODEL*MAX_SEQ_LEN)` | **absolute** `token*D_MODEL + col` |
| `wr_en`,`wr_addr`,`wr_data` | 1 / `X_AW` / `DATA_WIDTH` | **absolute** `token*D_MODEL + row` |
| `busy`,`done` | 1 | run handshake |

`attn_proj` drives its four internal `dense_layer`s with `w_base = 0`, so
`w_addr_*` is already the cache-relative address. It emits no layer term: the
caller applies the layer offset on both the weight read and the K/V writes.

---

## 1. ROM layout reminder (`transformer_engine.v:91-107`)

Per block, the four attention weight matrices and their bias vectors are
**interleaved**:

```
B_QW, B_QB, B_KW, B_KB, B_VW, B_VB, B_OW, B_OB
```

So we use **four** `weight_cache` instances (one per projection), each fed from
its own contiguous region, rather than one combined cache (which would need a
non-contiguous fill).

---

## 2. `hdl/attention.v` changes

### 2.1 Port list (`attention.v:32-37`)

Delete the byte-serial weight/bias load ports; add the four packed buses:

```verilog
// REMOVE:
//   input wire w_load,
//   input wire [W_ADDR_WIDTH-1:0] w_load_addr,
//   input wire signed [DATA_WIDTH-1:0] w_load_data,
//   input wire b_load,
//   input wire [B_ADDR_WIDTH-1:0] b_load_addr,
//   input wire signed [63:0] b_load_data,

// ADD (next to the parameters, and in the port list):
parameter NUM_PES = 1,
output wire [W_ADDR_WIDTH-1:0]       ap_w_addr_q, ap_w_addr_k, ap_w_addr_v, ap_w_addr_o,
output wire                          ap_w_gather_q, ap_w_gather_k, ap_w_gather_v, ap_w_gather_o,
input  wire signed [NUM_PES*DATA_WIDTH-1:0]
                                     ap_w_data_q, ap_w_data_k, ap_w_data_v, ap_w_data_o,
```

`D_MODEL_PES`/`X_AW` are derived inside `attn_proj`; pass
`.W_ADDR_WIDTH(W_ADDR_WIDTH)`, `.LEN_WIDTH(LEN_WIDTH)`, `.MAX_SEQ_LEN(MAX_SEQ_LEN)`.

### 2.2 Delete internal weight/bias storage (`attention.v:70-71, 82, 141, 146, 153-154, 211-218`)

Remove `weights[]`, `biases[]`, `weights_q`, `bias_q`, `weight_raddr`,
`bias_raddr`, `weights_we`, `biases_we`, and their two `always` blocks. Keep
`inputs[]`, `queries[]`, `k_cache[]`, `v_cache[]`, `context_data[]`, `scores[]`,
`exponentials[]`. Remove `projection_shift`, `proj_sum`, `proj_wdata`,
`proj_save`, `queries_we`, `kcache_we`, `vcache_we`, the `PROJ_MAC` product
case, and `proj_a`/`proj_b` (`proj_a` was only used by `PROJ_MAC`).

### 2.3 Instantiate `attn_proj`

```verilog
reg  [1:0]  proj_sel;
reg         proj_start;
wire        proj_done;

attn_proj #(
    .DATA_WIDTH(DATA_WIDTH), .ACC_WIDTH(ACC_WIDTH), .D_MODEL(D_MODEL),
    .NUM_PES(NUM_PES), .MAX_SEQ_LEN(MAX_SEQ_LEN),
    .Q_SHIFT(Q_SHIFT), .K_SHIFT(K_SHIFT), .V_SHIFT(V_SHIFT), .OUT_SHIFT(OUT_SHIFT),
    .W_ADDR_WIDTH(W_ADDR_WIDTH), .X_AW(X_ADDR_WIDTH), .LEN_WIDTH(LEN_WIDTH)
) proj (
    .clk(clk), .rst_n(rst_n),
    .start(proj_start), .projection(proj_sel),
    .length(seq_len_reg), .cache_len(cache_len),
    .w_addr_q(ap_w_addr_q), .w_addr_k(ap_w_addr_k),
    .w_addr_v(ap_w_addr_v), .w_addr_o(ap_w_addr_o),
    .w_gather_q(ap_w_gather_q), .w_gather_k(ap_w_gather_k),
    .w_gather_v(ap_w_gather_v), .w_gather_o(ap_w_gather_o),
    .w_data_q(ap_w_data_q), .w_data_k(ap_w_data_k),
    .w_data_v(ap_w_data_v), .w_data_o(ap_w_data_o),
    .x_addr(ap_x_addr), .x_data(ap_x_data),
    .wr_en(ap_wr_en), .wr_addr(ap_wr_addr), .wr_data(ap_wr_data),
    .busy(), .done(proj_done)
);
```

Add a registered `seq_len_reg` (or reuse `length`) so `attn_proj.length` is
stable throughout the Q/K/V/OUT runs, and use `X_ADDR_WIDTH` for `X_AW`.

### 2.4 Route activation reads

`attn_proj` emits the **absolute** address `token*D_MODEL + col`; `inputs` is
the source for Q/K/V and `context_data` for OUT. Both are synchronous RAMs, so
the registered read matches `dense_layer`'s one-cycle x address latency.

```verilog
wire [X_ADDR_WIDTH-1:0] ap_x_addr;
reg  signed [DATA_WIDTH-1:0] inputs_proj_q, context_proj_q;
always @(posedge clk) begin
    inputs_proj_q  <= inputs[ap_x_addr];
    context_proj_q <= context_data[ap_x_addr];
end
assign ap_x_data = (proj_sel == 2'd3) ? context_proj_q : inputs_proj_q;
```

The old `inputs_q`/`context_q` feeds stay for whatever remains, but the
projection path uses the pair above.

### 2.5 Route result writes

`ap_wr_addr` is already absolute `token*D_MODEL + row`. The K/V caches are
layer-stacked, so add the layer offset on the write side:

```verilog
always @(posedge clk) begin
    y_valid <= 1'b0;
    if (ap_wr_en) begin
        case (proj_sel)
            2'd0: queries[ap_wr_addr] <= ap_wr_data;
            2'd1: k_cache[layer_idx*MAX_SEQ_LEN*D_MODEL + ap_wr_addr] <= ap_wr_data;
            2'd2: v_cache[layer_idx*MAX_SEQ_LEN*D_MODEL + ap_wr_addr] <= ap_wr_data;
            2'd3: begin
                y_addr  <= ap_wr_addr;
                y_data  <= ap_wr_data;
                y_valid <= 1'b1;
            end
        endcase
    end
end
```

This is the same one-cycle output registration the old `PROJ_SAVE` path used,
so `transformer_engine.v`'s `attn_y_valid`/`attn_res_*` folding is unchanged.

### 2.6 New FSM sequencing

Replace `PROJ_READ/PROJ_MAC/PROJ_SAVE` (`attention.v:62-65, 279-310`) with a
projection runner. Keep `SCORE_*`, `EXP_*`, `VALUE_*`, `FINISH` untouched.

Concrete state encoding (replaces the old three projection states):

```verilog
localparam IDLE=0, PROJ_Q_START=1, PROJ_WAIT=2,
           PROJ_K_START=14, PROJ_V_START=15, PROJ_O_START=16,
           SCORE_READ=4, SCORE_MAC=5, SCORE_SAVE=6,
           EXP_READ=7, EXP=8, VALUE_READ=9, VALUE_MAC=10, VALUE_SAVE=11,
           VALUE_DIV=12, FINISH=13;
```

```verilog
IDLE: if (start) begin
    error <= 0;
    if (seq_len == 0 || seq_len > MAX_SEQ_LEN || cache_len > seq_len) begin
        error <= 1; done <= 1;
    end else begin
        length <= seq_len; seq_len_reg <= seq_len; busy <= 1;
        proj_sel <= 2'd0; proj_start <= 1'b0;
        token <= cache_len; row <= 0; col <= 0;
        accumulator <= 0; max_score <= 0;
        state <= PROJ_Q_START;
    end
end

PROJ_Q_START: begin proj_sel <= 2'd0; proj_start <= 1'b1; state <= PROJ_WAIT; end
PROJ_K_START: begin proj_sel <= 2'd1; proj_start <= 1'b1; state <= PROJ_WAIT; end
PROJ_V_START: begin proj_sel <= 2'd2; proj_start <= 1'b1; state <= PROJ_WAIT; end
PROJ_O_START: begin proj_sel <= 2'd3; proj_start <= 1'b1; state <= PROJ_WAIT; end

PROJ_WAIT: begin
    proj_start <= 1'b0;
    if (proj_done) begin
        case (proj_sel)
            2'd0: state <= PROJ_K_START;
            2'd1: state <= PROJ_V_START;
            2'd2: begin head <= 0; key_index <= 0; component <= 0;
                       accumulator <= 0; max_score <= 0; state <= SCORE_READ; end
            default: state <= FINISH;
        endcase
    end
end
```

In `VALUE_DIV` (`attention.v:344-361`), the final hand-off becomes:

```verilog
// was: projection <= 3; token <= cache_len; row <= 0; col <= 0; state <= PROJ_READ;
token <= cache_len; state <= PROJ_O_START;
```

`attn_proj` itself iterates every token in `[cache_len, length)` for each
projection, so the per-`token`/`row`/`col` projection loops in `attention.v`
are gone. `attn_proj` treats `cache_len >= length` as an empty run and pulses
`done`, so `cache_len == seq_len` no longer needs a special case here (the old
serial loop could spin; see Risks).

---

## 3. `hdl/transformer_engine.v` changes

### 3.1 Localparams

```verilog
localparam ATTN_ROWS = NUM_LAYERS * D_MODEL;   // rows per attention projection cache
```

Per projection: weights `NUM_LAYERS*D_MODEL` rows × `D_MODEL` inputs; biases
`NUM_LAYERS*D_MODEL`. (`ATTN_W = D_MODEL*D_MODEL` already exists.)

### 3.2 Four resident caches

```verilog
wire [W_ADDR_WIDTH-1:0] attn_ap_w_addr_q, attn_ap_w_addr_k,
                        attn_ap_w_addr_v, attn_ap_w_addr_o;
wire attn_ap_w_gather_q, attn_ap_w_gather_k, attn_ap_w_gather_v, attn_ap_w_gather_o;
wire [W_ADDR_WIDTH-1:0] aq_raddr, ak_raddr, av_raddr, ao_raddr;
wire aq_gather, ak_gather, av_gather, ao_gather;
wire signed [NUM_PES*DATA_WIDTH-1:0] aq_rdata, ak_rdata, av_rdata, ao_rdata;

// gather selects weights (stride D_MODEL*D_MODEL per layer) or biases (stride D_MODEL).
assign aq_raddr = aq_gather ? (l_cnt*(D_MODEL*D_MODEL) + attn_ap_w_addr_q)
                            : (l_cnt*D_MODEL          + attn_ap_w_addr_q);
assign ak_raddr = ak_gather ? (l_cnt*(D_MODEL*D_MODEL) + attn_ap_w_addr_k)
                            : (l_cnt*D_MODEL          + attn_ap_w_addr_k);
assign av_raddr = av_gather ? (l_cnt*(D_MODEL*D_MODEL) + attn_ap_w_addr_v)
                            : (l_cnt*D_MODEL          + attn_ap_w_addr_v);
assign ao_raddr = ao_gather ? (l_cnt*(D_MODEL*D_MODEL) + attn_ap_w_addr_o)
                            : (l_cnt*D_MODEL          + attn_ap_w_addr_o);

weight_cache #(.DATA_WIDTH(DATA_WIDTH), .NUM_PES(NUM_PES),
 .IN_FEATURES(D_MODEL), .OUT_ROWS(ATTN_ROWS), .ADDR_WIDTH(W_ADDR_WIDTH)) wc_aq (
 .clk(clk), .we_w(aq_we_w), .waddr(aq_waddr), .wdata(fill_wdata),
 .we_b(aq_we_b), .baddr(aq_baddr), .bdata(fill_wdata),
 .raddr(aq_raddr), .gather(aq_gather), .rdata(aq_rdata));
// ...identical for wc_ak (k), wc_av (v), wc_ao (o)
```

`attn_ap_w_addr_*` / `attn_ap_w_gather_*` come from the `attention` instance's
new outputs; `attn_ap_w_data_*` feed its new inputs.

Also add `.NUM_PES(NUM_PES), .W_ADDR_WIDTH(W_ADDR_WIDTH)` to the `attention`
instantiation so the `ap_w_addr_*` bus width matches the 32-bit cache
arithmetic (otherwise it defaults to `clog2(4*D_MODEL*D_MODEL)` and relies on
implicit truncation).

### 3.3 Extend the fill FSM (`transformer_engine.v:147-155, 433-481`)

The preload currently streams fc1/fc2 (4 phases). Grow it to 12 phases across
6 caches. Widen `fill_phase` to `[3:0]`/`[4:0]` and `fill_which` to 3 bits; add
per-cache write strobes. `fill_rom_addr` and `fill_caddr` stay single shared
buses that are decoded to the selected cache one cycle later (matching the
registered ROM read via `fill_we_d`/`fill_caddr_d` at `:465-481`).

| phase | data | ROM offset | cache `caddr` | `fill_limit` |
|---|---|---|---|---|
| 0 | fc1 w | `B_FC1W` | `layer*D_MODEL*D_FF + idx` | `D_MODEL*D_FF` |
| 1 | fc1 b | `B_FC1B` | `layer*D_FF + idx` | `D_FF` |
| 2 | fc2 w | `B_FC2W` | `layer*D_FF*D_MODEL + idx` | `D_FF*D_MODEL` |
| 3 | fc2 b | `B_FC2B` | `layer*D_MODEL + idx` | `D_MODEL` |
| 4 | Q w | `B_QW` | `layer*D_MODEL*D_MODEL + idx` | `D_MODEL*D_MODEL` |
| 5 | Q b | `B_QB` | `layer*D_MODEL + idx` | `D_MODEL` |
| 6 | K w | `B_KW` | `layer*D_MODEL*D_MODEL + idx` | `D_MODEL*D_MODEL` |
| 7 | K b | `B_KB` | `layer*D_MODEL + idx` | `D_MODEL` |
| 8 | V w | `B_VW` | `layer*D_MODEL*D_MODEL + idx` | `D_MODEL*D_MODEL` |
| 9 | V b | `B_VB` | `layer*D_MODEL + idx` | `D_MODEL` |
| 10 | O w | `B_OW` | `layer*D_MODEL*D_MODEL + idx` | `D_MODEL*D_MODEL` |
| 11 | O b | `B_OB` | `layer*D_MODEL + idx` | `D_MODEL` |

`fill_rom_addr = fill_blk_base + <ROM offset> + fill_idx` with
`fill_blk_base = TOK_SIZE + POS_SIZE + fill_layer*BLOCK_STRIDE` (unchanged).
`fill_which` selects `{fc1=0, fc2=1, aq=2, ak=3, av=4, ao=5}`. Write strobes:

```verilog
assign aq_we_w = fill_we_d && (fill_which_d == 3'd2) && !fill_bias_d;
assign aq_we_b = fill_we_d && (fill_which_d == 3'd2) &&  fill_bias_d;
// ...ak=3, av=4, ao=5
assign aq_waddr = fill_caddr_d;   // relative j*D_MODEL + i
assign aq_baddr = fill_caddr_d;   // relative j
```

The existing `caches_valid` one-time gate now covers the attention caches too,
so they are resident for every layer/token.

### 3.4 Remove per-token attention weight streaming

Delete `S_AW_ISS`, `S_AW_WAIT`, `S_AB_ISS`, `S_AB_WAIT`
(`transformer_engine.v:131-132, 606-637`) and the `attn_w_load`, `attn_w_addr`,
`attn_b_load`, `attn_b_addr`, `attn_w_data`, `attn_b_data` declarations
(`:182-193`). Trim the attention cases out of `ctrl_rom_addr` (`:304-310`), the
`rom_owner` decode, and the reset block (`:502-503`). Keep `S_ATTN_X`
(`:640-653`) to populate `attention.inputs`, and point `S_LN1_RUN`'s done
branch (`:597`) directly at `S_ATTN_X` instead of `S_AW_ISS`.

### 3.5 `attention` instantiation (`:334-348`)

```verilog
attention #(
    .DATA_WIDTH(DATA_WIDTH), .D_MODEL(D_MODEL), .NUM_HEADS(NUM_HEADS),
    .MAX_SEQ_LEN(MAX_SEQ_LEN), .NUM_LAYERS(NUM_LAYERS),
    .Q_SHIFT(Q_SHIFT), .K_SHIFT(K_SHIFT), .V_SHIFT(V_SHIFT), .OUT_SHIFT(OUT_SHIFT),
    .SCORE_MULT(SCORE_MULT), .SCORE_SHIFT(SCORE_SHIFT),
    .W_ADDR_WIDTH(W_ADDR_WIDTH), .NUM_PES(NUM_PES)
) attn_inst (
    .clk(clk), .rst_n(rst_n),
    .x_load(attn_x_load), .x_load_addr(attn_x_addr), .x_load_data(tmp_q),
    .ap_w_addr_q(attn_ap_w_addr_q), .ap_w_addr_k(attn_ap_w_addr_k),
    .ap_w_addr_v(attn_ap_w_addr_v), .ap_w_addr_o(attn_ap_w_addr_o),
    .ap_w_gather_q(attn_ap_w_gather_q), .ap_w_gather_k(attn_ap_w_gather_k),
    .ap_w_gather_v(attn_ap_w_gather_v), .ap_w_gather_o(attn_ap_w_gather_o),
    .ap_w_data_q(aq_rdata), .ap_w_data_k(ak_rdata),
    .ap_w_data_v(av_rdata), .ap_w_data_o(ao_rdata),
    .start(attn_start), .seq_len(attn_seq_len), .cache_len(attn_cache_len),
    .layer_idx(l_cnt[...]),
    .busy(attn_busy), .done(attn_done), .error(attn_error),
    .y_valid(attn_y_valid), .y_addr(attn_y_addr), .y_data(attn_y_data)
);
```

The `attention` side outputs raw stage-relative `attn_ap_w_addr_*`; **do not**
pre-add the layer offset in the connection — the offset is applied once, on the
`wc_a*` `raddr` in section 3.2.

---

## 4. Verification plan

1. `python3 -m unittest tests.test_attn_proj tests.test_dense_layer -v` — green
   (projection math, `NUM_PES` 1/2/4/8, `cache_len>0`); 10 tests in <3 s.
2. Add a `test_transformer_engine` case with `NUM_PES>1` comparing the full
   engine output against the `NUM_PES=1` golden, proving bit-exactness of the
   projection replacement (mirrors `test_dense_layer.test_scalar_matches_array`).
3. Re-run `tests/test_attention.py`; it drives the old `w_load/b_load` ports and
   must be updated in the same change to fill the four `wc_a*` caches instead.
4. Re-run `tests/test_throughput.py`; `attention_run` and `attn_weight_load`
   should drop, leaving MLP/LN as the top stages.
5. `build.tcl` + `quartus_sta` at `NUM_PES=4` to confirm area/Fmax still close
   (attention now holds 4 PE arrays = `4 × NUM_PES` DSPs).

---

## 5. Risks / open questions

- **Area:** attention now has 4 PE arrays; at `NUM_PES=4` that is 16 DSPs vs the
  current 1. Budget is 87 total, ~21 used, but watch the four 16 KB-class
  attention caches and the fitter.
- **`cache_len`/`length` width:** `attn_proj` uses `LEN_WIDTH = clog2(MAX_SEQ_LEN+1)`;
  pass the same `LEN_WIDTH` from `attention.v`.
- **Layer offset applied exactly once:** cache read (`l_cnt*D_MODEL*D_MODEL + ap_w_addr`
  for weights, `l_cnt*D_MODEL + ap_w_addr` for biases) and K/V write
  (`layer_idx*MAX_SEQ_LEN*D_MODEL + ap_wr_addr`). Do **not** double-apply it in
  the `attention` port connection.
- **Empty run:** `attn_proj` finishes immediately when `cache_len >= length`,
  which also makes the `cache_len == seq_len` corner safe; keep the existing
  `cache_len > seq_len` error guard.
- **Exactness:** integer MAC sum is order-independent, so parallelizing is
  bit-exact provided `dense_layer`'s post-sum shift/saturate boundary is kept —
  it is, and `attn_proj` tests confirm it across `NUM_PES`.
- **Reset/timing:** `attn_proj` needs `start` as a clean one-cycle pulse and
  `projection` stable for the whole run; both are handled by the `PROJ_*` states.