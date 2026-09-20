// Sequential LayerNorm over a length-N activation vector.
//
// y_i = sat( ((norm_i * gamma_i) >>> SHIFT) + beta_i )
// norm_i = ((x_i - mean) * inv) >>> SHIFT
// inv    = (1 << SHIFT) / isqrt(variance)   (1 << SHIFT when variance is 0)
//
// mean and variance are integer (truncating) averages over N elements, matching
// the controller's Python reference. gamma is a fixed-point INT value where
// 2**SHIFT represents 1.0; beta is already in output activation units. The
// module owns a single synchronous ROM port, time-multiplexed between gamma and
// beta, and reads x from an external buffer through a registered port.
//
// The integer divide and square root are multi-cycle restoring units so the
// datapath closes timing; the FSM waits for each before proceeding.
module layer_norm #(
    parameter DATA_WIDTH      = 8,
    parameter ACC_WIDTH       = 32,
    parameter N               = 64,
    parameter SHIFT           = 7,
    parameter X_ADDR_WIDTH    = (N <= 1) ? 1 : $clog2(N),
    parameter ROM_ADDR_WIDTH  = 16
) (
    input  wire                          clk,
    input  wire                          rst_n,
    input  wire                          start,

    output reg  [X_ADDR_WIDTH-1:0]       x_addr,
    input  wire signed [DATA_WIDTH-1:0]  x_data,

    output wire [ROM_ADDR_WIDTH-1:0]     rom_addr,
    input  wire signed [DATA_WIDTH-1:0]  rom_data,

    input  wire [ROM_ADDR_WIDTH-1:0]     g_base,
    input  wire [ROM_ADDR_WIDTH-1:0]     b_base,

    output reg                           y_wr_en,
    output reg  [X_ADDR_WIDTH-1:0]       y_addr,
    output reg  signed [DATA_WIDTH-1:0]  y_data,
    output reg                           done
);
    localparam S_IDLE    = 4'd0;
    localparam S_SUM     = 4'd1;
    localparam S_MEAN    = 4'd2;
    localparam S_VAR     = 4'd3;
    localparam S_VARDIV  = 4'd4;
    localparam S_SQRT    = 4'd5;
    localparam S_INVDIV  = 4'd6;
    localparam S_GAMMA   = 4'd7;
    localparam S_BETA    = 4'd8;
    localparam S_WRITE   = 4'd9;
    localparam S_SCALE   = 4'd10;
    localparam S_YWR     = 4'd11;
    localparam S_DONE    = 4'd12;

    localparam DW = 64;
    // d_val fits in DATA_WIDTH+1 signed bits; inv <= 2**SHIFT. Narrowing the
    // multiplier operands keeps the multiply off the critical path.
    localparam INV_W = SHIFT + 2;

    reg [3:0] state;
    reg [X_ADDR_WIDTH:0] i_cnt;
    reg signed [63:0] sum;
    reg signed [63:0] sumsq;
    reg signed [ACC_WIDTH-1:0]  mean;
    reg signed [ACC_WIDTH-1:0]  inv;
    reg signed [DATA_WIDTH-1:0] gamma_reg;
    reg signed [DATA_WIDTH:0]   norm_reg;
    reg signed [63:0]           scaled_reg;
    reg signed [DATA_WIDTH-1:0] beta_reg;
    reg signed [63:0] mean_num, var_num;
    reg signed [63:0] variance;
    reg signed [63:0] root;

    wire signed [DATA_WIDTH-1:0] x_val = x_data;
    wire signed [ACC_WIDTH-1:0]  x_ext = x_val;
    wire signed [ACC_WIDTH-1:0]  d_val = x_ext - mean;
    wire signed [DATA_WIDTH:0]   d_s = d_val;
    wire signed [INV_W-1:0]      inv_s = inv;
    wire signed [63:0] sum_next   = sum + x_ext;
    wire signed [2*DATA_WIDTH+1:0] d_sq = d_s * d_s;
    wire signed [63:0] sumsq_next = sumsq + d_sq;
    wire signed [DATA_WIDTH+INV_W:0] norm_prod = d_s * inv_s;
    wire signed [DATA_WIDTH:0]   norm = norm_prod >>> SHIFT;
    wire signed [2*DATA_WIDTH:0] scaled_prod = norm_reg * gamma_reg;
    wire signed [63:0] scaled = scaled_prod >>> SHIFT;

    // ── Sequential restoring divider (signed numerator, positive divisor) ──
    reg              div_start;
    reg signed [63:0] div_num, div_den;
    reg              div_busy;
    reg [DW:0]       div_rem;
    reg [DW-1:0]     div_q, div_ua, div_ub;
    reg              div_neg;
    reg [6:0]        div_cnt;
    wire signed [63:0] div_quot = div_neg ? -div_q : div_q;
    wire             div_done = div_busy && (div_cnt == 0);
    wire [DW:0]      div_shift = {div_rem[DW-1:0], div_ua[DW-1]};
    wire [DW:0]      div_ubx   = {1'b0, div_ub};
    wire             div_ge    = (div_shift >= div_ubx);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            div_busy <= 1'b0; div_cnt <= 0; div_rem <= 0; div_q <= 0;
            div_ua <= 0; div_ub <= 0; div_neg <= 1'b0;
        end else if (div_start) begin
            div_neg  <= div_num < 0;
            div_ua   <= div_num < 0 ? -div_num : div_num;
            div_ub   <= div_den;
            div_rem  <= 0;
            div_q    <= 0;
            // A zero divisor has no meaningful quotient; define it as 0 and
            // finish immediately so the FSM can never stall waiting for done.
            div_cnt  <= (div_den == 0) ? 7'd0 : DW;
            div_busy <= 1'b1;
        end else if (div_busy) begin
            if (div_cnt != 0) begin
                div_rem <= div_ge ? (div_shift - div_ubx) : div_shift;
                div_q   <= div_ge ? {div_q[DW-2:0], 1'b1} : {div_q[DW-2:0], 1'b0};
                div_ua  <= {div_ua[DW-2:0], 1'b0};
                div_cnt <= div_cnt - 1'b1;
            end else begin
                div_busy <= 1'b0;
            end
        end
    end

    // ── Sequential restoring square root (unsigned) ──
    reg              sqrt_start;
    reg [63:0]       sqrt_value;
    reg              sqrt_busy;
    reg [DW-1:0]     sqrt_v, sqrt_root;
    reg [DW+1:0]     sqrt_rem;
    reg [5:0]        sqrt_cnt;
    wire             sqrt_done = sqrt_busy && (sqrt_cnt == 0);
    wire [DW+1:0]    sqrt_shift = {sqrt_rem[DW-1:0], sqrt_v[DW-1:DW-2]};
    wire [DW+1:0]    sqrt_trial = {sqrt_root, 2'b01};
    wire             sqrt_ge = (sqrt_shift >= sqrt_trial);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            sqrt_busy <= 1'b0; sqrt_cnt <= 0; sqrt_v <= 0; sqrt_root <= 0; sqrt_rem <= 0;
        end else if (sqrt_start) begin
            sqrt_v    <= sqrt_value;
            sqrt_rem  <= 0;
            sqrt_root <= 0;
            sqrt_cnt  <= DW/2;
            sqrt_busy <= 1'b1;
        end else if (sqrt_busy) begin
            if (sqrt_cnt != 0) begin
                sqrt_rem  <= sqrt_ge ? (sqrt_shift - sqrt_trial) : sqrt_shift;
                sqrt_root <= sqrt_ge ? {sqrt_root[DW-2:0], 1'b1} : {sqrt_root[DW-2:0], 1'b0};
                sqrt_v    <= sqrt_v << 2;
                sqrt_cnt  <= sqrt_cnt - 1'b1;
            end else begin
                sqrt_busy <= 1'b0;
            end
        end
    end

    // gamma is addressed during S_GAMMA, beta during S_BETA.
    assign rom_addr = (state == S_GAMMA) ? (g_base + i_cnt)
                    : (state == S_BETA)  ? (b_base + i_cnt)
                    : {ROM_ADDR_WIDTH{1'b0}};

    function signed [DATA_WIDTH-1:0] saturate;
        input signed [63:0] value;
        begin
            if (value > ((1 << (DATA_WIDTH-1)) - 1))
                saturate = (1 << (DATA_WIDTH-1)) - 1;
            else if (value < -(1 << (DATA_WIDTH-1)))
                saturate = -(1 << (DATA_WIDTH-1));
            else
                saturate = value[DATA_WIDTH-1:0];
        end
    endfunction

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE;
            done <= 1'b0;
            y_wr_en <= 1'b0;
            x_addr <= 0;
            y_addr <= 0;
            y_data <= 0;
            i_cnt <= 0;
            sum <= 0;
            sumsq <= 0;
            mean <= 0;
            inv <= 0;
            gamma_reg <= 0;
            norm_reg <= 0; scaled_reg <= 0; beta_reg <= 0;
            mean_num <= 0; var_num <= 0; variance <= 0; root <= 0;
            div_start <= 1'b0; sqrt_start <= 1'b0;
            div_num <= 0; div_den <= N; sqrt_value <= 0;
        end else begin
            done <= 1'b0;
            y_wr_en <= 1'b0;
            div_start <= 1'b0;
            sqrt_start <= 1'b0;
            case (state)
                S_IDLE: begin
                    if (start) begin
                        i_cnt <= 0;
                        x_addr <= 1;
                        sum <= 0;
                        sumsq <= 0;
                        state <= S_SUM;
                    end
                end

                S_SUM: begin
                    sum <= sum + x_ext;
                    if (i_cnt == N-1) begin
                        mean_num <= sum_next;
                        div_num  <= sum_next;
                        div_den  <= N;
                        div_start <= 1'b1;
                        i_cnt <= 0;
                        x_addr <= 0;
                        state <= S_MEAN;
                    end else begin
                        i_cnt <= i_cnt + 1;
                        x_addr <= i_cnt + 2;
                    end
                end

                S_MEAN: begin
                    x_addr <= 0;
                    if (div_done) begin
                        mean <= div_quot[ACC_WIDTH-1:0];
                        x_addr <= 1;
                        state <= S_VAR;
                    end
                end

                S_VAR: begin
                    sumsq <= sumsq + d_val * d_val;
                    if (i_cnt == N-1) begin
                        var_num <= sumsq_next;
                        div_num <= sumsq_next;
                        div_den <= N;
                        div_start <= 1'b1;
                        i_cnt <= 0;
                        x_addr <= 0;
                        state <= S_VARDIV;
                    end else begin
                        i_cnt <= i_cnt + 1;
                        x_addr <= i_cnt + 2;
                    end
                end

                S_VARDIV: begin
                    if (div_done) begin
                        variance <= div_quot;
                        if (div_quot == 0) begin
                            inv <= (1 << SHIFT);
                            state <= S_GAMMA;
                        end else begin
                            sqrt_value <= div_quot;
                            sqrt_start <= 1'b1;
                            state <= S_SQRT;
                        end
                    end
                end

                S_SQRT: begin
                    if (sqrt_done) begin
                        root <= sqrt_root;
                        // Guard against a zero root (should not happen for a
                        // positive variance) so the inverse divide never sees a
                        // zero denominator and cannot stall.
                        if (sqrt_root == 0) begin
                            inv <= (1 << SHIFT);
                            state <= S_GAMMA;
                        end else begin
                            div_num <= (1 << SHIFT);
                            div_den <= sqrt_root;
                            div_start <= 1'b1;
                            state <= S_INVDIV;
                        end
                    end
                end

                S_INVDIV: begin
                    if (div_done) begin
                        inv <= div_quot[ACC_WIDTH-1:0];
                        state <= S_GAMMA;
                    end
                end

                S_GAMMA: state <= S_BETA;

                S_BETA: begin
                    gamma_reg <= rom_data;
                    state <= S_WRITE;
                end

                S_WRITE: begin
                    beta_reg <= rom_data;
                    norm_reg <= norm;
                    state <= S_SCALE;
                end

                S_SCALE: begin
                    scaled_reg <= scaled;
                    state <= S_YWR;
                end

                S_YWR: begin
                    y_wr_en <= 1'b1;
                    y_addr  <= i_cnt;
                    y_data  <= saturate(scaled_reg + beta_reg);
                    if (i_cnt == N-1) state <= S_DONE;
                    else begin
                        i_cnt <= i_cnt + 1;
                        x_addr <= i_cnt + 1;
                        state <= S_GAMMA;
                    end
                end

                S_DONE: begin
                    done <= 1'b1;
                    x_addr <= 0;
                    state <= S_IDLE;
                end

                default: state <= S_IDLE;
            endcase
        end
    end
endmodule
