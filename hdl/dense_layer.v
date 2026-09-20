// ═══════════════════════════════════════════════════════════════
// fpGPT — Time-Multiplexed Dense (Linear) Layer Engine
// Target: Intel Cyclone V
//
// Computes: y[j] = activation( (sum_i(W[j][i] * x[i]) + b[j]) >>> SHIFT_BITS )
//   for j = 0..OUT_FEATURES-1
//
// Weights and biases are read from a synchronous (1-cycle latency) ROM. The
// base addresses are runtime inputs so one instance can serve every layer in
// the network. x is read from an external activation buffer through a
// synchronous (1-cycle latency) port.
//
// ── Output-stationary PE array ──
// NUM_PES output features are computed concurrently, one MAC per PE per cycle.
// The weight ROM must deliver NUM_PES packed signed bytes per read:
//
//     w_data[k*DATA_WIDTH +: DATA_WIDTH] = value for lane k
//
// Two gather modes select how the per-lane addresses are derived from the
// single `w_addr` group base:
//
//     w_gather = 1 : lane k address = w_addr + k*IN_FEATURES   (weight matrix)
//     w_gather = 0 : lane k address = w_addr + k               (bias vector)
//
// With NUM_PES = 1 this module is bit-for-bit the original scalar engine and
// the ROM interface is exactly one DATA_WIDTH-wide word. OUT_FEATURES must be
// a multiple of NUM_PES.
// ═══════════════════════════════════════════════════════════════

module dense_layer #(
    parameter DATA_WIDTH    = 8,
    parameter ACC_WIDTH     = 32,
    parameter IN_FEATURES   = 64,
    parameter OUT_FEATURES  = 64,
    parameter SHIFT_BITS    = 7,
    parameter ACTIVATION    = "relu",    // "relu", "gelu", "none"
    parameter W_ADDR_WIDTH  = 16,
    parameter HAS_BIAS      = 1,
    parameter NUM_PES       = 1          // output features computed per pass
) (
    input  wire                    clk,
    input  wire                    rst_n,
    input  wire                    start,       // Pulse high to begin computation
    output reg                     done,        // High for 1 cycle when complete

    // Runtime weight/bias base addresses in the unified ROM
    input  wire [W_ADDR_WIDTH-1:0] w_base,
    input  wire [W_ADDR_WIDTH-1:0] b_base,

    // Weight ROM interface (synchronous read, NUM_PES packed bytes)
    output wire [W_ADDR_WIDTH-1:0] w_addr,
    output wire                    w_gather,
    input  wire signed [NUM_PES*DATA_WIDTH-1:0] w_data,

    // Input activation buffer (synchronous read, broadcast to all lanes)
    output wire [((IN_FEATURES <= 1) ? 1 : $clog2(IN_FEATURES))-1:0] x_addr,
    input  wire signed [DATA_WIDTH-1:0] x_data,

    // Output activation buffer
    output reg                                   y_wr_en,
    output wire [((OUT_FEATURES <= 1) ? 1 : $clog2(OUT_FEATURES))-1:0] y_addr,
    output wire signed [DATA_WIDTH-1:0]          y_data
);
    localparam IN_AW  = (IN_FEATURES <= 1)  ? 1 : $clog2(IN_FEATURES);
    localparam OUT_AW = (OUT_FEATURES <= 1) ? 1 : $clog2(OUT_FEATURES);
    localparam PES_AW = (NUM_PES <= 1)      ? 1 : $clog2(NUM_PES);

    // synthesis translate_off
    initial begin
        if (IN_FEATURES < 1 || OUT_FEATURES < 1)
            $fatal(1, "dense_layer: feature counts must be positive");
        if (NUM_PES < 1 || OUT_FEATURES % NUM_PES != 0)
            $fatal(1, "dense_layer: OUT_FEATURES must be a multiple of NUM_PES");
    end
    // synthesis translate_on

    // ── FSM States ──
    localparam S_IDLE     = 3'd0;
    localparam S_LOAD     = 3'd1;   // Clear accumulators, present W[j_base][0]
    localparam S_COMPUTE  = 3'd2;   // MAC loop over IN_FEATURES
    localparam S_BIAS     = 3'd3;   // Add bias (if present)
    localparam S_DRAIN    = 3'd4;   // Flush the last pipelined product
    localparam S_WRITE    = 3'd5;   // Write the NUM_PES outputs
    localparam S_NEXT     = 3'd6;   // Advance to the next output group
    localparam S_DONE     = 3'd7;

    reg [2:0] state;

    // Counters
    reg [IN_AW:0]  i_cnt;    // Input dimension counter
    reg [OUT_AW:0] j_base;   // First output feature of the current group
    reg [PES_AW:0] lane_cnt; // Output write lane within the group

    // MAC control (shared by all lanes; timing matches the scalar engine)
    wire mac_clear  = (state == S_LOAD);
    wire mac_enable = (state == S_COMPUTE && i_cnt >= 1) ||
                      (state == S_BIAS) || (state == S_DRAIN);
    wire signed [DATA_WIDTH-1:0] mac_act = (state == S_BIAS)   ? {{(DATA_WIDTH-1){1'b0}}, 1'b1}
                                         : (state == S_DRAIN)  ? {DATA_WIDTH{1'b0}}
                                         : x_data;

    // Packed activation outputs, one slice per lane.
    wire signed [NUM_PES*DATA_WIDTH-1:0] act_packed;

    // ── PE array: one 2-stage MAC + activation per output lane ──
    genvar k;
    generate
        for (k = 0; k < NUM_PES; k = k + 1) begin : gen_pe
            wire signed [DATA_WIDTH-1:0] w_lane = w_data[k*DATA_WIDTH +: DATA_WIDTH];
            wire signed [ACC_WIDTH-1:0]  acc_lane;

            mac_unit #(
                .DATA_WIDTH(DATA_WIDTH),
                .ACC_WIDTH(ACC_WIDTH)
            ) mac_inst (
                .clk(clk),
                .rst_n(rst_n),
                .clear(mac_clear),
                .enable(mac_enable),
                .weight(w_lane),
                .activation(mac_act),
                .accumulator(acc_lane)
            );

            activation #(
                .DATA_WIDTH(DATA_WIDTH),
                .ACC_WIDTH(ACC_WIDTH),
                .SHIFT_BITS(SHIFT_BITS),
                .MODE(ACTIVATION)
            ) act_inst (
                .acc_in(acc_lane),
                .data_out(act_packed[k*DATA_WIDTH +: DATA_WIDTH])
            );
        end
    endgenerate

    // Select the lane being written. y_wr_en gates the value, so the mux may
    // settle during non-write cycles without effect.
    assign y_data = act_packed[lane_cnt*DATA_WIDTH +: DATA_WIDTH];

    // y_addr tracks lane_cnt combinationally so the address and y_data always
    // describe the same lane; y_wr_en is the registered write strobe.
    assign y_addr = (state == S_WRITE) ? (j_base[OUT_AW-1:0] + lane_cnt[PES_AW-1:0])
                                       : {OUT_AW{1'b0}};

    // Combinational ROM/buffer addresses. The synchronous ROM turns the address
    // presented this cycle into data on the next cycle.
    assign w_addr = (state == S_LOAD)    ? (w_base + j_base * IN_FEATURES)
                  : (state == S_COMPUTE) ? (((i_cnt == IN_FEATURES) && HAS_BIAS)
                                             ? (b_base + j_base)
                                             : (w_base + j_base * IN_FEATURES + i_cnt))
                  : (state == S_BIAS)    ? (b_base + j_base)
                  : {W_ADDR_WIDTH{1'b0}};

    // Weight rows are strided by IN_FEATURES; bias values are contiguous.
    assign w_gather = (state == S_LOAD)
                   || (state == S_COMPUTE && !((i_cnt == IN_FEATURES) && HAS_BIAS));

    // The activation buffer is a synchronous (1-cycle latency) RAM, so present
    // the next input index now; the registered read delivers x[i_cnt-1] in time
    // for the MAC that consumes it.
    assign x_addr = (state == S_COMPUTE) ? i_cnt[IN_AW-1:0] : {IN_AW{1'b0}};

    // ── Main FSM ──
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state      <= S_IDLE;
            done       <= 1'b0;
            i_cnt      <= 0;
            j_base     <= 0;
            lane_cnt   <= 0;
            y_wr_en    <= 1'b0;
        end else begin
            done       <= 1'b0;
            y_wr_en    <= 1'b0;

            case (state)
                S_IDLE: begin
                    if (start) begin
                        j_base <= 0;
                        state  <= S_LOAD;
                    end
                end

                S_LOAD: begin
                    i_cnt <= 0;
                    state <= S_COMPUTE;
                end

                S_COMPUTE: begin
                    if (i_cnt < IN_FEATURES) begin
                        i_cnt <= i_cnt + 1;
                    end else if (HAS_BIAS) begin
                        state <= S_BIAS;
                    end else begin
                        state <= S_DRAIN;
                    end
                end

                S_BIAS: begin
                    state <= S_DRAIN;
                end

                S_DRAIN: begin
                    lane_cnt <= 0;
                    y_wr_en  <= 1'b1;   // write lane 0 during the first S_WRITE
                    state    <= S_WRITE;
                end

                S_WRITE: begin
                    if (lane_cnt == NUM_PES - 1) begin
                        y_wr_en <= 1'b0;
                        state   <= S_NEXT;
                    end else begin
                        lane_cnt <= lane_cnt + 1'b1;
                        y_wr_en  <= 1'b1;
                    end
                end

                S_NEXT: begin
                    if (j_base + NUM_PES < OUT_FEATURES) begin
                        j_base <= j_base + NUM_PES;
                        state  <= S_LOAD;
                    end else begin
                        state <= S_DONE;
                    end
                end

                S_DONE: begin
                    done  <= 1'b1;
                    state <= S_IDLE;
                end

                default: state <= S_IDLE;
            endcase
        end
    end

endmodule
