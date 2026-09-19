// ═══════════════════════════════════════════════════════════════
// fpGPT — Time-Multiplexed Dense (Linear) Layer Engine
// Target: Intel Cyclone V
//
// Computes: y[j] = activation( sum_i(W[j][i] * x[i]) + b[j] )
//   for j = 0..OUT_FEATURES-1
//
// Architecture: Uses a single MAC unit that iterates over the
// input dimension (IN_FEATURES cycles per output element).
// Total cycles per layer = IN_FEATURES * OUT_FEATURES + overhead.
//
// This is the "compact" variant — uses 1 DSP block per layer.
// For higher throughput, instantiate NUM_PES parallel MAC units
// to compute multiple output elements simultaneously.
// ═══════════════════════════════════════════════════════════════

module dense_layer #(
    parameter DATA_WIDTH    = 8,
    parameter ACC_WIDTH     = 32,
    parameter IN_FEATURES   = 64,
    parameter OUT_FEATURES  = 64,
    parameter SHIFT_BITS    = 7,
    parameter ACTIVATION    = "relu",    // "relu", "gelu", "none"
    parameter NUM_PES       = 1,         // Number of parallel MAC units
    parameter W_ADDR_WIDTH  = 16,
    parameter W_BASE_ADDR   = 0,         // Base address in weight ROM
    parameter B_BASE_ADDR   = 0,         // Base address for bias in ROM
    parameter HAS_BIAS      = 1
) (
    input  wire                    clk,
    input  wire                    rst_n,
    input  wire                    start,       // Pulse high to begin computation
    output reg                     done,        // High for 1 cycle when complete

    // Weight ROM interface
    output reg  [W_ADDR_WIDTH-1:0] w_addr,     // Address to weight ROM
    input  wire signed [DATA_WIDTH-1:0] w_data, // Data from weight ROM (1-cycle latency)

    // Input activation buffer (dual-port RAM or register file)
    output reg  [$clog2(IN_FEATURES)-1:0] x_addr,
    input  wire signed [DATA_WIDTH-1:0]   x_data,

    // Output activation buffer
    output reg                                   y_wr_en,
    output reg  [$clog2(OUT_FEATURES)-1:0]       y_addr,
    output wire signed [DATA_WIDTH-1:0]          y_data
);

    // ── FSM States ──
    localparam S_IDLE     = 3'd0;
    localparam S_LOAD     = 3'd1;   // Pre-fetch first weight (ROM has 1-cycle latency)
    localparam S_COMPUTE  = 3'd2;   // MAC loop over IN_FEATURES
    localparam S_BIAS     = 3'd3;   // Add bias (if present)
    localparam S_ACTIVATE = 3'd4;   // Apply activation & write output
    localparam S_NEXT     = 3'd5;   // Advance to next output neuron
    localparam S_DONE     = 3'd6;

    reg [2:0] state;

    // Counters
    reg [$clog2(IN_FEATURES):0]  i_cnt;   // Input dimension counter
    reg [$clog2(OUT_FEATURES):0] j_cnt;   // Output dimension counter

    // MAC unit
    wire signed [ACC_WIDTH-1:0] acc;
    reg mac_clear, mac_enable;

    mac_unit #(
        .DATA_WIDTH(DATA_WIDTH),
        .ACC_WIDTH(ACC_WIDTH)
    ) mac_inst (
        .clk(clk),
        .rst_n(rst_n),
        .clear(mac_clear),
        .enable(mac_enable),
        .weight(w_data),
        .activation(x_data),
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

    // ── Main FSM ──
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state      <= S_IDLE;
            done       <= 1'b0;
            i_cnt      <= 0;
            j_cnt      <= 0;
            w_addr     <= 0;
            x_addr     <= 0;
            y_addr     <= 0;
            y_wr_en    <= 1'b0;
            mac_clear  <= 1'b0;
            mac_enable <= 1'b0;
        end else begin
            // Defaults
            done       <= 1'b0;
            y_wr_en    <= 1'b0;
            mac_clear  <= 1'b0;
            mac_enable <= 1'b0;

            case (state)
                S_IDLE: begin
                    if (start) begin
                        j_cnt <= 0;
                        state <= S_LOAD;
                    end
                end

                S_LOAD: begin
                    // Clear accumulator for new output neuron
                    mac_clear <= 1'b1;
                    i_cnt     <= 0;
                    // Pre-fetch: set ROM address for W[j][0]
                    w_addr <= W_BASE_ADDR + j_cnt * IN_FEATURES;
                    x_addr <= 0;
                    state  <= S_COMPUTE;
                end

                S_COMPUTE: begin
                    // ROM data arrives this cycle (from address set last cycle)
                    mac_enable <= 1'b1;

                    if (i_cnt < IN_FEATURES - 1) begin
                        i_cnt  <= i_cnt + 1;
                        w_addr <= W_BASE_ADDR + j_cnt * IN_FEATURES + i_cnt + 1;
                        x_addr <= i_cnt + 1;
                    end else begin
                        // Done with this output neuron's dot product
                        if (HAS_BIAS) begin
                            w_addr <= B_BASE_ADDR + j_cnt;
                            state  <= S_BIAS;
                        end else begin
                            state <= S_ACTIVATE;
                        end
                    end
                end

                S_BIAS: begin
                    // Bias arrives from ROM; add it to accumulator
                    // (Treat bias as weight * 1, where activation = 1 = 01h)
                    // Actually, we'll just add the bias value directly
                    mac_enable <= 1'b1;
                    // x_data should be 1 for bias addition — handled by controller
                    state <= S_ACTIVATE;
                end

                S_ACTIVATE: begin
                    // Activation function is combinational on acc
                    // Write result to output buffer
                    y_wr_en <= 1'b1;
                    y_addr  <= j_cnt[$clog2(OUT_FEATURES)-1:0];
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
