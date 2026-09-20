// ═══════════════════════════════════════════════════════════════
// fpGPT — Parallel attention projection engine
//
// Runs one of the four self-attention projections (Q, K, V or output) for a
// range of tokens using the output-stationary PE array in hdl/dense_layer.v.
//
// Each projection has its own packed weight bus (hdl/weight_cache.v contract),
// because the compiler interleaves the four weight matrices with their bias
// vectors in the unified ROM and a single combined cache would need a
// non-contiguous fill. For projection p the cache is read as:
//
//     gather = 1 : raddr = layer*D_MODEL^2 + j_base*D_MODEL + i
//                  byte k = W_p[layer][j_base + k][i]
//     gather = 0 : raddr = layer*D_MODEL + j_base
//                  byte k = b_p[layer][j_base + k]
//
// `x_addr` is the *absolute* activation address token*D_MODEL + col, so the
// caller can route it to either the layer input buffer or the context buffer
// (projection 3). Results stream out one lane per cycle with the absolute
// address token*D_MODEL + row, tagged only by the caller's knowledge of the
// projection currently running.
//
// This is the parallel replacement for the serial PROJ_READ/PROJ_MAC/PROJ_SAVE
// loop in hdl/attention.v; it is kept separate so it can be verified before
// integration.
// ═══════════════════════════════════════════════════════════════

module attn_proj #(
    parameter DATA_WIDTH   = 8,
    parameter ACC_WIDTH    = 32,
    parameter D_MODEL      = 64,
    parameter NUM_PES      = 1,
    parameter MAX_SEQ_LEN  = 64,
    parameter Q_SHIFT      = 7,
    parameter K_SHIFT      = 7,
    parameter V_SHIFT      = 7,
    parameter OUT_SHIFT    = 7,
    parameter W_ADDR_WIDTH = 16,
    parameter X_AW         = (D_MODEL*MAX_SEQ_LEN <= 1) ? 1 : $clog2(D_MODEL*MAX_SEQ_LEN),
    parameter LEN_WIDTH    = $clog2(MAX_SEQ_LEN+1)
) (
    input  wire                          clk,
    input  wire                          rst_n,
    input  wire                          start,        // pulse to begin one projection
    input  wire [1:0]                    projection,   // 0=Q 1=K 2=V 3=OUT
    input  wire [LEN_WIDTH-1:0]          length,       // sequence length (exclusive end)
    input  wire [LEN_WIDTH-1:0]          cache_len,    // first token to process

    // Packed weight ROMs, one per projection (hdl/weight_cache.v contract)
    output wire [W_ADDR_WIDTH-1:0]       w_addr_q, w_addr_k, w_addr_v, w_addr_o,
    output wire                          w_gather_q, w_gather_k, w_gather_v, w_gather_o,
    input  wire signed [NUM_PES*DATA_WIDTH-1:0] w_data_q, w_data_k, w_data_v, w_data_o,

    // Activation buffer (synchronous read, broadcast to all lanes)
    output wire [X_AW-1:0]               x_addr,
    input  wire signed [DATA_WIDTH-1:0]  x_data,

    // Result writes: absolute address token*D_MODEL + row
    output wire                          wr_en,
    output wire [X_AW-1:0]               wr_addr,
    output wire signed [DATA_WIDTH-1:0]  wr_data,

    output reg                           busy,
    output reg                           done
);
    localparam IN_AW  = (D_MODEL <= 1) ? 1 : $clog2(D_MODEL);
    localparam ROWS   = D_MODEL;                 // output features per projection

    localparam S_IDLE     = 2'd0,
               S_TOK_START = 2'd1,
               S_TOK_WAIT  = 2'd2,
               S_FINISH    = 2'd3;

    reg [1:0]              state;
    reg [LEN_WIDTH-1:0]    token;
    reg                    dl_start;

    // ── Four PE-array projections, time-multiplexed (one active at a time) ──
    wire dl_q_start = dl_start && (projection == 2'd0);
    wire dl_k_start = dl_start && (projection == 2'd1);
    wire dl_v_start = dl_start && (projection == 2'd2);
    wire dl_o_start = dl_start && (projection == 2'd3);

    wire [IN_AW-1:0]        x_addr_q, x_addr_k, x_addr_v, x_addr_o;
    wire                    y_we_q, y_we_k, y_we_v, y_we_o;
    wire [IN_AW-1:0]        y_addr_q, y_addr_k, y_addr_v, y_addr_o;
    wire signed [DATA_WIDTH-1:0] y_data_q, y_data_k, y_data_v, y_data_o;
    wire                    done_q, done_k, done_v, done_o;

    dense_layer #(
        .DATA_WIDTH(DATA_WIDTH), .ACC_WIDTH(ACC_WIDTH),
        .IN_FEATURES(D_MODEL), .OUT_FEATURES(ROWS), .SHIFT_BITS(Q_SHIFT),
        .ACTIVATION("none"), .W_ADDR_WIDTH(W_ADDR_WIDTH), .HAS_BIAS(1),
        .NUM_PES(NUM_PES)
    ) proj_q (
        .clk(clk), .rst_n(rst_n), .start(dl_q_start), .done(done_q),
        .w_base(0), .b_base(0),
        .w_addr(w_addr_q), .w_gather(w_gather_q), .w_data(w_data_q),
        .x_addr(x_addr_q), .x_data(x_data),
        .y_wr_en(y_we_q), .y_addr(y_addr_q), .y_data(y_data_q)
    );

    dense_layer #(
        .DATA_WIDTH(DATA_WIDTH), .ACC_WIDTH(ACC_WIDTH),
        .IN_FEATURES(D_MODEL), .OUT_FEATURES(ROWS), .SHIFT_BITS(K_SHIFT),
        .ACTIVATION("none"), .W_ADDR_WIDTH(W_ADDR_WIDTH), .HAS_BIAS(1),
        .NUM_PES(NUM_PES)
    ) proj_k (
        .clk(clk), .rst_n(rst_n), .start(dl_k_start), .done(done_k),
        .w_base(0), .b_base(0),
        .w_addr(w_addr_k), .w_gather(w_gather_k), .w_data(w_data_k),
        .x_addr(x_addr_k), .x_data(x_data),
        .y_wr_en(y_we_k), .y_addr(y_addr_k), .y_data(y_data_k)
    );

    dense_layer #(
        .DATA_WIDTH(DATA_WIDTH), .ACC_WIDTH(ACC_WIDTH),
        .IN_FEATURES(D_MODEL), .OUT_FEATURES(ROWS), .SHIFT_BITS(V_SHIFT),
        .ACTIVATION("none"), .W_ADDR_WIDTH(W_ADDR_WIDTH), .HAS_BIAS(1),
        .NUM_PES(NUM_PES)
    ) proj_v (
        .clk(clk), .rst_n(rst_n), .start(dl_v_start), .done(done_v),
        .w_base(0), .b_base(0),
        .w_addr(w_addr_v), .w_gather(w_gather_v), .w_data(w_data_v),
        .x_addr(x_addr_v), .x_data(x_data),
        .y_wr_en(y_we_v), .y_addr(y_addr_v), .y_data(y_data_v)
    );

    dense_layer #(
        .DATA_WIDTH(DATA_WIDTH), .ACC_WIDTH(ACC_WIDTH),
        .IN_FEATURES(D_MODEL), .OUT_FEATURES(ROWS), .SHIFT_BITS(OUT_SHIFT),
        .ACTIVATION("none"), .W_ADDR_WIDTH(W_ADDR_WIDTH), .HAS_BIAS(1),
        .NUM_PES(NUM_PES)
    ) proj_o (
        .clk(clk), .rst_n(rst_n), .start(dl_o_start), .done(done_o),
        .w_base(0), .b_base(0),
        .w_addr(w_addr_o), .w_gather(w_gather_o), .w_data(w_data_o),
        .x_addr(x_addr_o), .x_data(x_data),
        .y_wr_en(y_we_o), .y_addr(y_addr_o), .y_data(y_data_o)
    );

    // ── Active-projection mux (activation read and result write) ──
    reg [IN_AW-1:0]        x_addr_mux, y_addr_mux;
    reg                    y_we_mux;
    reg signed [DATA_WIDTH-1:0] y_data_mux;
    reg                    done_mux;

    always @(*) begin
        case (projection)
            2'd0: begin
                x_addr_mux = x_addr_q;
                y_we_mux = y_we_q; y_addr_mux = y_addr_q; y_data_mux = y_data_q; done_mux = done_q;
            end
            2'd1: begin
                x_addr_mux = x_addr_k;
                y_we_mux = y_we_k; y_addr_mux = y_addr_k; y_data_mux = y_data_k; done_mux = done_k;
            end
            2'd2: begin
                x_addr_mux = x_addr_v;
                y_we_mux = y_we_v; y_addr_mux = y_addr_v; y_data_mux = y_data_v; done_mux = done_v;
            end
            default: begin
                x_addr_mux = x_addr_o;
                y_we_mux = y_we_o; y_addr_mux = y_addr_o; y_data_mux = y_data_o; done_mux = done_o;
            end
        endcase
    end

    assign x_addr  = token * D_MODEL + x_addr_mux;
    assign wr_en   = y_we_mux;
    assign wr_addr = token * D_MODEL + y_addr_mux;
    assign wr_data = y_data_mux;

    // ── Token sequencing FSM ──
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state    <= S_IDLE;
            token    <= 0;
            dl_start <= 1'b0;
            busy     <= 1'b0;
            done     <= 1'b0;
        end else begin
            done <= 1'b0;
            case (state)
                S_IDLE: begin
                    if (start) begin
                        busy  <= 1'b1;
                        token <= cache_len;
                        state <= (cache_len >= length) ? S_FINISH : S_TOK_START;
                    end
                end

                S_TOK_START: begin
                    dl_start <= 1'b1;
                    state    <= S_TOK_WAIT;
                end

                S_TOK_WAIT: begin
                    dl_start <= 1'b0;
                    if (done_mux) begin
                        if (token == length - 1) begin
                            state <= S_FINISH;
                        end else begin
                            token <= token + 1'b1;
                            state <= S_TOK_START;
                        end
                    end
                end

                S_FINISH: begin
                    done  <= 1'b1;
                    busy  <= 1'b0;
                    state <= S_IDLE;
                end

                default: state <= S_IDLE;
            endcase
        end
    end

endmodule
