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
// beta, and reads x from an external buffer through a combinational port.
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
    localparam S_IDLE   = 3'd0;
    localparam S_SUM    = 3'd1;
    localparam S_VAR    = 3'd2;
    localparam S_GAMMA  = 3'd3;
    localparam S_BETA   = 3'd4;
    localparam S_WRITE  = 3'd5;
    localparam S_DONE   = 3'd6;

    reg [2:0] state;
    reg [X_ADDR_WIDTH:0] i_cnt;
    reg signed [63:0] sum;
    reg signed [63:0] sumsq;
    reg signed [ACC_WIDTH-1:0]  mean;
    reg signed [ACC_WIDTH-1:0]  inv;
    reg signed [DATA_WIDTH-1:0] gamma_reg;

    wire signed [DATA_WIDTH-1:0] x_val = x_data;
    wire signed [ACC_WIDTH-1:0]  x_ext = x_val;
    wire signed [ACC_WIDTH-1:0]  d_val = x_ext - mean;
    wire signed [63:0] sum_next   = sum + x_ext;
    wire signed [63:0] sumsq_next = sumsq + d_val * d_val;
    wire signed [63:0] d64 = d_val;
    wire signed [63:0] inv64 = inv;
    wire signed [63:0] norm = (d64 * inv64) >>> SHIFT;
    wire signed [63:0] g64 = gamma_reg;
    wire signed [63:0] scaled = (norm * g64) >>> SHIFT;

    // gamma is addressed during S_GAMMA, beta during S_BETA.
    assign rom_addr = (state == S_GAMMA) ? (g_base + i_cnt)
                    : (state == S_BETA)  ? (b_base + i_cnt)
                    : {ROM_ADDR_WIDTH{1'b0}};

    // Integer floor square root.
    function [ACC_WIDTH-1:0] isqrt;
        input [ACC_WIDTH-1:0] value;
        reg [ACC_WIDTH-1:0] v, r, b;
        begin
            v = value;
            r = {ACC_WIDTH{1'b0}};
            b = {{(ACC_WIDTH-1){1'b0}}, 1'b1};
            while (b <= (v >> 2)) b = b << 2;
            while (b != 0) begin
                if (v >= r + b) begin
                    v = v - (r + b);
                    r = (r >> 1) + b;
                end else begin
                    r = r >> 1;
                end
                b = b >> 2;
            end
            isqrt = r;
        end
    endfunction

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
        end else begin
            done <= 1'b0;
            y_wr_en <= 1'b0;
            case (state)
                S_IDLE: begin
                    if (start) begin
                        i_cnt <= 0;
                        x_addr <= 0;
                        sum <= 0;
                        sumsq <= 0;
                        state <= S_SUM;
                    end
                end

                S_SUM: begin
                    sum <= sum + x_ext;
                    if (i_cnt == N-1) begin
                        mean <= sum_next / N;
                        i_cnt <= 0;
                        x_addr <= 0;
                        state <= S_VAR;
                    end else begin
                        i_cnt <= i_cnt + 1;
                        x_addr <= i_cnt + 1;
                    end
                end

                S_VAR: begin
                    sumsq <= sumsq + d_val * d_val;
                    if (i_cnt == N-1) begin
                        i_cnt <= 0;
                        x_addr <= 0;
                        if ((sumsq_next / N) == 0) inv <= (1 << SHIFT);
                        else inv <= (1 << SHIFT) / isqrt(sumsq_next / N);
                        state <= S_GAMMA;
                    end else begin
                        i_cnt <= i_cnt + 1;
                        x_addr <= i_cnt + 1;
                    end
                end

                S_GAMMA: state <= S_BETA;

                S_BETA: begin
                    gamma_reg <= rom_data;
                    state <= S_WRITE;
                end

                S_WRITE: begin
                    y_wr_en <= 1'b1;
                    y_addr  <= i_cnt;
                    y_data  <= saturate(scaled + rom_data);
                    if (i_cnt == N-1) state <= S_DONE;
                    else begin
                        i_cnt <= i_cnt + 1;
                        x_addr <= i_cnt + 1;
                        state <= S_GAMMA;
                    end
                end

                S_DONE: begin
                    done <= 1'b1;
                    state <= S_IDLE;
                end

                default: state <= S_IDLE;
            endcase
        end
    end
endmodule
