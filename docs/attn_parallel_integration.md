# Parallel attention integration — draft patch plan

**Status: NOT APPLIED.** This is a reviewable plan for replacing the serial
projection loop in `hdl/attention.v` with `hdl/attn_proj.v`, plus the resident
attention weight caches it needs in `hdl/transformer_engine.v`.

Prereqs already landed (verified standalone):
- `hdl/attn_proj.v` — 4 time-multiplexed `dense_layer` PE arrays, one packed
  weight bus per projection, per-projection shift/bias.
- `hdl/weight_cache.v` — packed/banked resident cache.
- `tests/test_attn_proj.py` — bit-exact vs reference, NUM_PES 1/2/4.

Owner: whoever is out of `attention.v` / `transformer_engine.v` when this lands.
Do **not** apply while a Quartus build is mid-flight.

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

### 2.1 Port list
Remove the byte-serial weight/bias load ports; add the four packed buses:

```verilog
// REMOVE:
//   input wire w_load,
//   input wire [W_ADDR_WIDTH-1:0] w_load_addr,
//   input wire signed [DATA_WIDTH-1:0] w_load_data,
//   input wire b_load,
//   input wire [B_ADDR_WIDTH-1:0] b_load_addr,
//   input wire signed [63:0] b_load_data,

// ADD:
parameter NUM_PES = 1,
output wire [W_ADDR_WIDTH-1:0] ap_w_addr_q, ap_w_addr_k, ap_w_addr_v, ap_w_addr_o,
output wire                    ap_w_gather_q, ap_w_gather_k, ap_w_gather_v, ap_w_gather_o,
input  wire signed [NUM_PES*DATA_WIDTH-1:0]
                               ap_w_data_q, ap_w_data_k, ap_w_data_v, ap_w_data_o,
```

### 2.2 Delete internal weight/bias storage
Remove `weights[]`, `biases[]`, their write/read `always` blocks, `weights_q`,
`bias_q`, `weight_raddr`, `bias_raddr`, `biases_we`, `weights_we`. Keep
`inputs[]`, `queries[]`, `k_cache[]`, `v_cache[]`, `context_data[]`, `scores[]`,
`exponentials[]`.

### 2.3 Instantiate `attn_proj`
```verilog
attn_proj #(
    .DATA_WIDTH(DATA_WIDTH), .ACC_WIDTH(ACC_WIDTH), .D_MODEL(D_MODEL),
    .NUM_PES(NUM_PES), .MAX_SEQ_LEN(MAX_SEQ_LEN),
    .Q_SHIFT(Q_SHIFT), .K_SHIFT(K_SHIFT), .V_SHIFT(V_SHIFT), .OUT_SHIFT(OUT_SHIFT),
    .W_ADDR_WIDTH(W_ADDR_WIDTH)
) proj (
    .clk(clk), .rst_n(rst_n),
    .start(proj_start), .projection(proj_sel),
    .length(length), .cache_len(cache_len_reg),
    .w_addr_q(ap_w_addr_q), .w_addr_k(ap_w_addr_k),
    .w_addr_v(ap_w_addr_v), .w_addr_o(ap_w_addr_o),
    .w_gather_q(ap_w_gather_q), .w_gather_k(ap_w_gather_k),
    .w_gather_v(ap_w_gather_v), .w_gather_o(ap_w_gather_o),
    .w_data_q(ap_w_data_q), .w_data_k(ap_w_data_k),
    .w_data_v(ap_w_data_v), .w_data_o(ap_w_data_o),
    .x_addr(proj_x_addr), .x_data(proj_x_data),
    .wr_en(proj_wr_en), .wr_addr(proj_wr_addr), .wr_data(proj_wr_data),
    .busy(), .done(proj_done)
);
```

### 2.4 Route activation reads (absolute address from `attn_proj`)
```verilog
// attn_proj emits token*D_MODEL + col; inputs is used for Q/K/V, context for OUT.
reg signed [DATA_WIDTH-1:0] inputs_q, context_q;
always @(posedge clk) begin
    inputs_q  <= inputs[proj_x_addr];
    context_q <= context_data[proj_x_addr];
end
assign proj_x_data = (proj_sel == 2'd3) ? context_q : inputs_q;
```

### 2.5 Route result writes
```verilog
always @(posedge clk) begin
    if (proj_wr_en && proj_sel == 2'd0) queries[proj_wr_addr] <= proj_wr_data;
    if (proj_wr_en && proj_sel == 2'd1)
        k_cache[layer_idx*MAX_SEQ_LEN*D_MODEL + proj_wr_addr] <= proj_wr_data;
    if (proj_wr_en && proj_sel == 2'd2)
        v_cache[layer_idx*MAX_SEQ_LEN*D_MODEL + proj_wr_addr] <= proj_wr_data;
    // OUT stream (registered one cycle, matching the old PROJ_SAVE y path)
    y_valid <= 1'b0;
    if (proj_wr_en && proj_sel == 2'd3) begin
        y_addr  <= proj_wr_addr;
        y_data  <= proj_wr_data;
        y_valid <= 1'b1;
    end
end
```

### 2.6 New FSM sequencing
Replace the `PROJ_READ/PROJ_MAC/PROJ_SAVE` states with a projection runner:

```
IDLE -> (start) latch length/cache_len_reg
     -> PROJ_Q_START -> PROJ_WAIT
     -> PROJ_K_START -> PROJ_WAIT
     -> PROJ_V_START -> PROJ_WAIT
     -> SCORE_READ .. VALUE_DIV            (unchanged)
     -> PROJ_O_START -> PROJ_WAIT
     -> FINISH
```

- `proj_start` pulses for one cycle in each `PROJ_x_START`.
- `PROJ_WAIT`: wait for `proj_done`.
- `proj_sel` is a registered 2-bit value set in the start states.
- The existing score/exp/value/div states and counters (`head`, `key_index`,
  `component`, `token`) stay as-is; only the projection phase is replaced.

---

## 3. `hdl/transformer_engine.v` changes

### 3.1 Localparams
```verilog
localparam ATTN_ROWS = NUM_LAYERS * D_MODEL;   // rows per projection cache
```
(weights per projection = `D_MODEL*D_MODEL`; biases = `D_MODEL`.)

### 3.2 Four caches
```verilog
wire [W_ADDR_WIDTH-1:0] aq_raddr, ak_raddr, av_raddr, ao_raddr;
wire aq_gather, ak_gather, av_gather, ao_gather;
wire signed [NUM_PES*DATA_WIDTH-1:0] aq_rdata, ak_rdata, av_rdata, ao_rdata;

// layer offset: raddr = l_cnt*D_MODEL*D_MODEL + ap_w_addr (gather)
//               raddr = l_cnt*D_MODEL          + ap_w_addr (bias)
assign aq_raddr = aq_gather ? (l_cnt*(D_MODEL*D_MODEL) + attn_ap_w_addr_q)
                            : (l_cnt*D_MODEL          + attn_ap_w_addr_q);
// ...same for k/v/o

weight_cache #(.DATA_WIDTH(DATA_WIDTH), .NUM_PES(NUM_PES),
 .IN_FEATURES(D_MODEL), .OUT_ROWS(ATTN_ROWS), .ADDR_WIDTH(W_ADDR_WIDTH)) wc_q (
 .clk(clk), .we_w(aq_we_w), .waddr(aq_waddr), .wdata(fill_wdata),
 .we_b(aq_we_b), .baddr(aq_baddr), .bdata(fill_wdata),
 .raddr(aq_raddr), .gather(aq_gather), .rdata(aq_rdata));
// ...k/v/o
```
`attn_ap_w_addr_*` come from the `attention` instance's `ap_w_addr_*` outputs.

### 3.3 Extend the fill FSM
`fill_phase` grows to 4 bits with 12 phases:

| phase | data | ROM base | cache caddr |
|---|---|---|---|
| 0 | fc1 w | `B_FC1W` | `layer*D_MODEL*D_FF + idx` |
| 1 | fc1 b | `B_FC1B` | `layer*D_FF + idx` |
| 2 | fc2 w | `B_FC2W` | `layer*D_FF*D_MODEL + idx` |
| 3 | fc2 b | `B_FC2B` | `layer*D_MODEL + idx` |
| 4 | Q w | `B_QW` | `layer*D_MODEL*D_MODEL + idx` |
| 5 | Q b | `B_QB` | `layer*D_MODEL + idx` |
| 6 | K w | `B_KW` | … |
| 7 | K b | `B_KB` | … |
| 8 | V w | `B_VW` | … |
| 9 | V b | `B_VB` | … |
| 10 | O w | `B_OW` | … |
| 11 | O b | `B_OB` | … |

`fill_limit` = `D_MODEL*D_FF` / `D_FF` / `D_FF*D_MODEL` / `D_MODEL` for the
fc phases, `D_MODEL*D_MODEL` for the `*w` attn phases, `D_MODEL` for `*b`.
`fill_which` widens to select `{fc1, fc2, q, k, v, o}`; `fill_is_bias` stays.

### 3.4 Remove per-token attention weight streaming
Delete `S_AW_ISS`, `S_AW_WAIT`, `S_AB_ISS`, `S_AB_WAIT`, `attn_w_load`,
`attn_w_addr`, `attn_b_load`, `attn_b_addr`, `attn_w_data`, `attn_b_data`, and
the `rom_owner` attention cases. The `S_ATTN_X` activation load stays (still
needed to populate `attention.inputs`).

### 3.5 `attention` instantiation
Replace the `w_load/b_load/...` connections with the four packed buses, and add
`.NUM_PES(NUM_PES)`.

---

## 4. Verification plan

1. `tests/test_attn_proj.py` — already green (projection math, per-cache).
2. Add a `test_transformer_engine` case with `NUM_PES>1` that compares the full
   engine output against the `NUM_PES=1` golden, proving bit-exactness of the
   projection replacement (mirrors `test_packed_pe_array_matches_scalar`).
3. Re-run `tests/test_attention.py` unchanged — it exercises `attention.v`
   through the old interface; it will need its weight/bias load stimulus
   replaced by cache fills, so update it in the same change.
4. Re-run `tests/test_throughput.py` and expect the profile to move:
   `attention_run` and `attn_weight_load` should drop; MLP/LN become the top
   stages.
5. `build.tcl` + `quartus_sta` at `NUM_PES=4` to confirm area/Fmax still close
   (4× more DSPs in attention: 4 projections × NUM_PES).

---

## 5. Risks / open questions

- **Area:** attention now has 4 PE arrays; at NUM_PES=4 that is 16 DSPs vs the
  current 1. Budget is 87 total, ~21 used, so it fits, but watch the fitter.
- **`cache_len`/`length` width:** `attn_proj` uses `LEN_WIDTH = clog2(MAX_SEQ_LEN+1)`;
  make sure `attention.v` passes matching widths.
- **Layer offset for OUT projection:** `attn_proj` has no layer term; the
  `l_cnt*…` offset must be applied on the cache read (section 3.2) and the
  k/v write offset on the write side (section 2.5). Both must use `layer_idx`.
- **Reset/timing:** `attn_proj` assumes `length > cache_len`; `attention` must
  guard the empty case (already does for invalid lengths).
- **Exactness:** integer MAC sum is order-independent, so parallelizing is
  bit-exact provided `dense_layer`'s post-sum shift/saturate boundary is kept —
  it is.
