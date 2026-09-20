// ═══════════════════════════════════════════════════════════════
// fpGPT — Resident banked weight cache (M10K)
//
// Holds a stage's weights and biases on-chip in NUM_PES banks so a single
// synchronous read returns NUM_PES packed bytes, matching the packed bus of
// hdl/dense_layer.v (NUM_PES PEs):
//
//     gather = 1 (weights): byte k = W[j_base + k][i]
//     gather = 0 (biases) : byte k = b[j_base + k]
//
// with a read address `raddr` relative to the stage base:
//     weights: raddr = j_base*IN_FEATURES + i
//     biases : raddr = j_base
//
// Weights are bank-interleaved by output row (bank b holds row j for every
// j % NUM_PES == b); biases are bank-interleaved by byte (bank b holds bias j
// for every j % NUM_PES == b). IN_FEATURES and NUM_PES must be powers of two.
//
// The flat fill ports let the engine stream the unified ROM image in once; the
// cache is then read NUM_PES-wide on every layer/token without re-streaming.
//
// NOTE: the bank memories live in the weight_bank submodule and are only
// *instantiated* in the generate loop. Declaring the memory arrays directly
// inside a generate block makes Quartus 25.1 Analysis & Synthesis hang/OOM
// (observed: >8 min with no progress on a single 64 KB RAM), whereas the
// identical memory outside the generate infers in seconds.
// ═══════════════════════════════════════════════════════════════

module weight_bank #(
    parameter DATA_WIDTH  = 8,
    parameter WBANK_DEPTH = 16384,
    parameter WBANK_AW    = 14,
    parameter BBANK_DEPTH = 256,
    parameter BBANK_AW    = 8
) (
    input  wire                              clk,

    input  wire                              we_w,
    input  wire [WBANK_AW-1:0]               waddr,
    input  wire signed [DATA_WIDTH-1:0]      wdata,
    input  wire [WBANK_AW-1:0]               wr_raddr,

    input  wire                              we_b,
    input  wire [BBANK_AW-1:0]               baddr,
    input  wire signed [DATA_WIDTH-1:0]      bdata,
    input  wire [BBANK_AW-1:0]               br_raddr,

    output reg  signed [DATA_WIDTH-1:0]      wq,
    output reg  signed [DATA_WIDTH-1:0]      bq
);
    (* ramstyle = "M10K" *) reg [DATA_WIDTH-1:0] wmem [0:WBANK_DEPTH-1];
    (* ramstyle = "M10K" *) reg [DATA_WIDTH-1:0] bmem [0:BBANK_DEPTH-1];

    always @(posedge clk) begin
        if (we_w)
            wmem[waddr] <= wdata;
        wq <= wmem[wr_raddr];
    end

    always @(posedge clk) begin
        if (we_b)
            bmem[baddr] <= bdata;
        bq <= bmem[br_raddr];
    end
endmodule


module weight_cache #(
    parameter DATA_WIDTH  = 8,
    parameter NUM_PES     = 1,
    parameter IN_FEATURES = 64,     // power of two
    parameter OUT_ROWS    = 256,    // capacity in output features
    parameter ADDR_WIDTH  = 16
) (
    input  wire                                  clk,

    // Weight fill: waddr = j*IN_FEATURES + i
    input  wire                                  we_w,
    input  wire [ADDR_WIDTH-1:0]                 waddr,
    input  wire signed [DATA_WIDTH-1:0]          wdata,

    // Bias fill: baddr = j
    input  wire                                  we_b,
    input  wire [ADDR_WIDTH-1:0]                 baddr,
    input  wire signed [DATA_WIDTH-1:0]          bdata,

    // Packed read: raddr relative to the stage base, gather selects the layout
    input  wire [ADDR_WIDTH-1:0]                 raddr,
    input  wire                                  gather,
    output wire signed [NUM_PES*DATA_WIDTH-1:0]  rdata
);
    localparam P_IDX_W    = (NUM_PES <= 1) ? 1 : $clog2(NUM_PES);
    localparam BANK_ROWS  = (OUT_ROWS + NUM_PES - 1) / NUM_PES;
    localparam WBANK_DEPTH = BANK_ROWS * IN_FEATURES;
    localparam WBANK_AW    = (WBANK_DEPTH <= 1) ? 1 : $clog2(WBANK_DEPTH);
    localparam BBANK_DEPTH = BANK_ROWS;
    localparam BBANK_AW    = (BBANK_DEPTH <= 1) ? 1 : $clog2(BBANK_DEPTH);

    // synthesis translate_off
    initial begin
        if (IN_FEATURES < 1)
            $fatal(1, "weight_cache: IN_FEATURES must be positive");
        if (NUM_PES < 1)
            $fatal(1, "weight_cache: NUM_PES must be positive");
    end
    // synthesis translate_on

    // ── Weight fill translation: flat f = j*IN + i ──
    wire [ADDR_WIDTH:0] w_j     = waddr / IN_FEATURES;
    wire [ADDR_WIDTH:0] w_i     = waddr % IN_FEATURES;
    wire [P_IDX_W-1:0]  wr_bank = w_j % NUM_PES;
    wire [WBANK_AW-1:0] wr_addr = (w_j / NUM_PES) * IN_FEATURES + w_i;

    // ── Bias fill translation: flat j ──
    wire [P_IDX_W-1:0]  br_bank = baddr % NUM_PES;
    wire [BBANK_AW-1:0] br_addr = baddr / NUM_PES;

    // ── Read translation ──
    wire [ADDR_WIDTH:0] r_j      = raddr / IN_FEATURES;
    wire [ADDR_WIDTH:0] r_i      = raddr % IN_FEATURES;
    wire [WBANK_AW-1:0] wr_raddr = (r_j / NUM_PES) * IN_FEATURES + r_i;
    wire [BBANK_AW-1:0] br_raddr = raddr / NUM_PES;

    // gather is consumed one cycle after the read address, matching the
    // registered read latency of the two bank memories.
    reg gather_d;
    always @(posedge clk) gather_d <= gather;

    genvar b;
    generate
        for (b = 0; b < NUM_PES; b = b + 1) begin : bank
            wire signed [DATA_WIDTH-1:0] wq, bq;

            weight_bank #(
                .DATA_WIDTH(DATA_WIDTH),
                .WBANK_DEPTH(WBANK_DEPTH), .WBANK_AW(WBANK_AW),
                .BBANK_DEPTH(BBANK_DEPTH), .BBANK_AW(BBANK_AW)
            ) u_bank (
                .clk(clk),
                .we_w(we_w && (wr_bank == b)), .waddr(wr_addr), .wdata(wdata),
                .wr_raddr(wr_raddr),
                .we_b(we_b && (br_bank == b)), .baddr(br_addr), .bdata(bdata),
                .br_raddr(br_raddr),
                .wq(wq), .bq(bq)
            );

            assign rdata[b*DATA_WIDTH +: DATA_WIDTH] = gather_d ? wq : bq;
        end
    endgenerate

endmodule
