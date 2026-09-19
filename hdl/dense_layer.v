// ═══════════════════════════════════════════════════════════════
// fpGPT — Time-Multiplexed Dense (Linear) Layer Engine
// Target: Intel Cyclone V
//
// Computes: y[j] = activation( (sum_i(W[j][i] * x[i]) + b[j]) >>> SHIFT_BITS )
//   for j = 0..OUT_FEATURES-1
//
// Weights and biases are read from a synchronous (1-cycle latency) ROM. The
// base addresses are runtime inputs so one instance can serve every layer in
// the network. x is read combinationally from an external activation buffer.
//
// This is the "compact" variant — uses 1 DSP block per instance.
// ═══════════════════════════════════════════════════════════════

module dense_layer #(
    parameter DATA_WIDTH    = 8,
    parameter ACC_WIDTH     = 32,
    parameter IN_FEATURES   = 64,
    parameter OUT_FEATURES  = 64,
    parameter SHIFT_BITS    = 7,
    parameter ACTIVATION    = "relu",    // "relu", "gelu", "none"
    parameter W_ADDR_WIDTH  = 16,
    parameter HAS_BIAS      = 1
) (
    input  wire                    clk,
    input  wire                    rst_n,
    input  wire                    start,       // Pulse high to begin computation
    output reg                     done,        // High for 1 cycle when complete

    // Runtime weight/bias base addresses in the unified ROM
    input  wire [W_ADDR_WIDTH-1:0] w_base,
    input  wire [W_ADDR_WIDTH-1:0] b_base,

    // Weight ROM interface (synchronous read, 1-cycle latency)
    output wire [W_ADDR_WIDTH-1:0] w_addr,
    input  wire signed [DATA_WIDTH-1:0] w_data,

    // Input activation buffer (combinational read)
    output wire [((IN_FEATURES <= 1) ? 1 : $clog2(IN_FEATURES))-1:0] x_addr,
    input  wire signed [DATA_WIDTH-1:0] x_data,

    // Output activation buffer
    output reg                                   y_wr_en,
    output reg  [((OUT_FEATURES <= 1) ? 1 : $clog2(OUT_FEATURES))-1:0] y_addr,
    output wire signed [DATA_WIDTH-1:0]          y_data
);
    localparam IN_AW  = (IN_FEATURES <= 1)  ? 1 : $clog2(IN_FEATURES);
    localparam OUT_AW = (OUT_FEATURES <= 1) ? 1 : $clog2(OUT_FEATURES);

    // ── FSM States ──
    localparam S_IDLE     = 3'd0;
    localparam S_LOAD     = 3'd1;   // Clear accumulator, present W[j][0]
    localparam S_COMPUTE  = 3'd2;   // MAC loop over IN_FEATURES
    localparam S_BIAS     = 3'd3;   // Add bias (if present)
    localparam S_ACTIVATE = 3'd4;   // Apply activation & write output
    localparam S_NEXT     = 3'd5;   // Advance to next output neuron
    localparam S_DONE     = 3'd6;

    reg [2:0] state;

    // Counters
    reg [IN_AW:0]  i_cnt;   // Input dimension counter
    reg [OUT_AW:0] j_cnt;   // Output dimension counter

    // MAC unit
    wire signed [ACC_WIDTH-1:0] acc;
    wire mac_clear  = (state == S_LOAD);
    wire mac_enable = (state == S_COMPUTE && i_cnt >= 1) || (state == S_BIAS);
    wire signed [DATA_WIDTH-1:0] mac_act = (state == S_BIAS) ? {{(DATA_WIDTH-1){1'b0}}, 1'b1} : x_data;

    mac_unit #(
        .DATA_WIDTH(DATA_WIDTH),
        .ACC_WIDTH(ACC_WIDTH)
    ) mac_inst (
        .clk(clk),
        .rst_n(rst_n),
        .clear(mac_clear),
        .enable(mac_enable),
        .weight(w_data),
        .activation(mac_act),
        .accumulator(acc)
    );

    // Activation function
    activation #(
        .DATA_WIDTH(DATA_WIDTH),
        .ACC_WIDTH(ACC_WIDTH),
        .SHIFT_BITS(SHIFT_BITS),
        .MODE(ACTIVATION)
    ) act_inst (
        .acc_in(acc),
        .data_out(y_data)
    );

    // Combinational ROM/buffer addresses. The synchronous ROM turns the address
    // presented this cycle into data on the next cycle.
    assign w_addr = (state == S_LOAD)    ? (w_base + j_cnt * IN_FEATURES)
                  : (state == S_COMPUTE) ? (((i_cnt == IN_FEATURES) && HAS_BIAS)
                                             ? (b_base + j_cnt)
                                             : (w_base + j_cnt * IN_FEATURES + i_cnt))
                  : (state == S_BIAS)    ? (b_base + j_cnt)
                  : {W_ADDR_WIDTH{1'b0}};

    assign x_addr = (state == S_COMPUTE && i_cnt != 0) ? (i_cnt - 1) : {IN_AW{1'b0}};

    // ── Main FSM ──
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state      <= S_IDLE;
            done       <= 1'b0;
            i_cnt      <= 0;
            j_cnt      <= 0;
            y_addr     <= 0;
            y_wr_en    <= 1'b0;
        end else begin
            done       <= 1'b0;
            y_wr_en    <= 1'b0;

            case (state)
                S_IDLE: begin
                    if (start) begin
                        j_cnt <= 0;
                        state <= S_LOAD;
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
                        state <= S_ACTIVATE;
                    end
                end

                S_BIAS: begin
                    state <= S_ACTIVATE;
                end

                S_ACTIVATE: begin
                    y_wr_en <= 1'b1;
                    y_addr  <= j_cnt[OUT_AW-1:0];
                    state   <= S_NEXT;
                end

                S_NEXT: begin
                    if (j_cnt < OUT_FEATURES - 1) begin
                        j_cnt <= j_cnt + 1;
                        state <= S_LOAD;
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
