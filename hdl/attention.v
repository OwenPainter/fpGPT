// Sequential causal multi-head self-attention, including Q/K/V/out projections.
// See hdl/attention.md for the loading protocol and fixed-point contract.
module attention #(
    parameter DATA_WIDTH = 8,
    parameter D_MODEL = 64,
    parameter NUM_HEADS = 4,
    parameter MAX_SEQ_LEN = 64,
    parameter Q_SHIFT = 7,
    parameter K_SHIFT = 7,
    parameter V_SHIFT = 7,
    parameter OUT_SHIFT = 7,
    // round(256 / sqrt(D_MODEL / NUM_HEADS)); default head dimension = 16
    parameter SCORE_MULT = 64,
    // Q/K integer products -> score units of 1/4 (includes multiplier precision).
    // Defaults assume Q and K each represent integers divided by 2.
    parameter SCORE_SHIFT = 8,
    parameter X_ADDR_WIDTH = (D_MODEL*MAX_SEQ_LEN <= 1) ? 1 : $clog2(D_MODEL*MAX_SEQ_LEN),
    parameter W_ADDR_WIDTH = (4*D_MODEL*D_MODEL <= 1) ? 1 : $clog2(4*D_MODEL*D_MODEL),
    parameter B_ADDR_WIDTH = (4*D_MODEL <= 1) ? 1 : $clog2(4*D_MODEL),
    parameter LEN_WIDTH = $clog2(MAX_SEQ_LEN+1)
) (
    input wire clk, rst_n,
    input wire x_load,
    input wire [X_ADDR_WIDTH-1:0] x_load_addr,
    input wire signed [DATA_WIDTH-1:0] x_load_data,
    input wire w_load,
    input wire [W_ADDR_WIDTH-1:0] w_load_addr,
    input wire signed [DATA_WIDTH-1:0] w_load_data,
    input wire b_load,
    input wire [B_ADDR_WIDTH-1:0] b_load_addr,
    input wire signed [63:0] b_load_data,
    input wire start,
    input wire [LEN_WIDTH-1:0] seq_len,
    output reg busy, done, error,
    output reg y_valid,
    output reg [X_ADDR_WIDTH-1:0] y_addr,
    output reg signed [DATA_WIDTH-1:0] y_data
);
    localparam HEAD_DIM = D_MODEL / NUM_HEADS;
    // synthesis translate_off
    initial begin
        if (D_MODEL < 1 || NUM_HEADS < 1 || D_MODEL % NUM_HEADS != 0 || MAX_SEQ_LEN < 1)
            $fatal(1, "attention: invalid dimensions");
        if (DATA_WIDTH < 2 || DATA_WIDTH > 16 || SCORE_MULT < 1 ||
            Q_SHIFT < -31 || K_SHIFT < -31 || V_SHIFT < -31 || OUT_SHIFT < -31 ||
            Q_SHIFT > 62 || K_SHIFT > 62 || V_SHIFT > 62 || OUT_SHIFT > 62 ||
            SCORE_SHIFT < 0 || SCORE_SHIFT > 62)
            $fatal(1, "attention: invalid fixed-point parameters");
    end
    // synthesis translate_on
    localparam IDLE=0, PROJ_READ=1, PROJ_MAC=2, PROJ_SAVE=3,
               SCORE_READ=4, SCORE_MAC=5, SCORE_SAVE=6,
               EXP=7, VALUE_READ=8, VALUE_MAC=9, VALUE_SAVE=10,
               FINISH=11;
    reg [3:0] state;
    reg signed [DATA_WIDTH-1:0] inputs [0:MAX_SEQ_LEN*D_MODEL-1];
    reg signed [DATA_WIDTH-1:0] weights [0:4*D_MODEL*D_MODEL-1];
    reg signed [63:0] biases [0:4*D_MODEL-1];
    reg signed [DATA_WIDTH-1:0] queries [0:MAX_SEQ_LEN*D_MODEL-1];
    reg signed [DATA_WIDTH-1:0] keys [0:MAX_SEQ_LEN*D_MODEL-1];
    reg signed [DATA_WIDTH-1:0] values [0:MAX_SEQ_LEN*D_MODEL-1];
    reg signed [DATA_WIDTH-1:0] context_data [0:MAX_SEQ_LEN*D_MODEL-1];
    reg signed [63:0] scores [0:MAX_SEQ_LEN-1];
    reg [15:0] exponentials [0:MAX_SEQ_LEN-1];
    integer length, projection, token, row, col, head, key_index, component;
    reg signed [DATA_WIDTH-1:0] operand_a, operand_b;
    wire signed [2*DATA_WIDTH-1:0] product = operand_a * operand_b;
    reg signed [63:0] accumulator, max_score;
    wire signed [63:0] scaled_score = (accumulator * SCORE_MULT) >>> SCORE_SHIFT;
    wire signed [63:0] delta = max_score - scores[key_index];
    reg [15:0] exp_operand;
    reg signed [63:0] denominator;
    wire signed [DATA_WIDTH+16:0] weighted_value = operand_a * $signed({1'b0, exp_operand});
    integer projection_shift;
    always @(*) begin
        case (projection)
            0: projection_shift = Q_SHIFT;
            1: projection_shift = K_SHIFT;
            2: projection_shift = V_SHIFT;
            default: projection_shift = OUT_SHIFT;
        endcase
    end

    function signed [DATA_WIDTH-1:0] saturate;
        input signed [63:0] value;
        begin
            if (value > ((64'sd1 << (DATA_WIDTH-1))-1))
                saturate = (64'sd1 << (DATA_WIDTH-1))-1;
            else if (value < -(64'sd1 << (DATA_WIDTH-1)))
                saturate = -(64'sd1 << (DATA_WIDTH-1));
            else saturate = value[DATA_WIDTH-1:0];
        end
    endfunction

    function signed [63:0] requant;
        input signed [63:0] value;
        input integer bits;
        begin
            if (bits >= 0) requant = value >>> bits;
            else requant = value <<< -bits;
        end
    endfunction

    // round(32768 * exp(-d/4)), with a zero tail beyond d=32.
    function [15:0] exp_lut;
        input signed [63:0] d;
        begin
            case (d)
                0: exp_lut=32768; 1: exp_lut=25520; 2: exp_lut=19875;
                3: exp_lut=15479; 4: exp_lut=12055; 5: exp_lut=9388;
                6: exp_lut=7312; 7: exp_lut=5694; 8: exp_lut=4435;
                9: exp_lut=3454; 10: exp_lut=2690; 11: exp_lut=2095;
                12: exp_lut=1631; 13: exp_lut=1271; 14: exp_lut=990;
                15: exp_lut=771; 16: exp_lut=600; 17: exp_lut=467;
                18: exp_lut=364; 19: exp_lut=283; 20: exp_lut=221;
                21: exp_lut=172; 22: exp_lut=134; 23: exp_lut=104;
                24: exp_lut=81; 25: exp_lut=63; 26: exp_lut=49;
                27: exp_lut=38; 28: exp_lut=30; 29: exp_lut=23;
                30: exp_lut=18; 31: exp_lut=14; 32: exp_lut=11;
                default: exp_lut=0;
            endcase
        end
    endfunction

    // Registered reads keep the matrix memories compatible with synchronous RAM.
    always @(posedge clk) begin
        if (!busy && rst_n) begin
            if (x_load && x_load_addr < MAX_SEQ_LEN*D_MODEL) inputs[x_load_addr] <= x_load_data;
            if (w_load && w_load_addr < 4*D_MODEL*D_MODEL) weights[w_load_addr] <= w_load_data;
            if (b_load && b_load_addr < 4*D_MODEL) biases[b_load_addr] <= b_load_data;
        end
        if (state == PROJ_READ) begin
            if (projection == 3) operand_a <= context_data[token*D_MODEL+col];
            else operand_a <= inputs[token*D_MODEL+col];
            operand_b <= weights[projection*D_MODEL*D_MODEL+row*D_MODEL+col];
        end
        if (state == SCORE_READ) begin
            operand_a <= queries[token*D_MODEL+head*HEAD_DIM+component];
            operand_b <= keys[key_index*D_MODEL+head*HEAD_DIM+component];
        end
        if (state == VALUE_READ) begin
            operand_a <= values[key_index*D_MODEL+head*HEAD_DIM+component];
            exp_operand <= exponentials[key_index];
        end
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE; busy <= 0; done <= 0; error <= 0;
            y_valid <= 0; y_addr <= 0; y_data <= 0;
            length <= 0; projection <= 0; token <= 0; row <= 0; col <= 0;
            head <= 0; key_index <= 0; component <= 0;
            accumulator <= 0; max_score <= 0; denominator <= 0;
        end else begin
            done <= 0; y_valid <= 0;
            case (state)
                IDLE: if (start) begin
                    error <= 0;
                    if (seq_len == 0 || seq_len > MAX_SEQ_LEN) begin
                        error <= 1; done <= 1;
                    end else begin
                        length <= seq_len; busy <= 1; projection <= 0;
                        token <= 0; row <= 0; col <= 0;
                        accumulator <= 0; state <= PROJ_READ;
                    end
                end
                PROJ_READ: state <= PROJ_MAC;
                PROJ_MAC: begin
                    accumulator <= accumulator + product;
                    if (col == D_MODEL-1) state <= PROJ_SAVE;
                    else begin col <= col+1; state <= PROJ_READ; end
                end
                PROJ_SAVE: begin
                    case (projection)
                        0: queries[token*D_MODEL+row] <= saturate(requant(accumulator+biases[row], projection_shift));
                        1: keys[token*D_MODEL+row] <= saturate(requant(accumulator+biases[D_MODEL+row], projection_shift));
                        2: values[token*D_MODEL+row] <= saturate(requant(accumulator+biases[2*D_MODEL+row], projection_shift));
                        3: begin
                            y_data <= saturate(requant(accumulator+biases[3*D_MODEL+row], projection_shift));
                            y_addr <= token*D_MODEL+row; y_valid <= 1;
                        end
                    endcase
                    accumulator <= 0; col <= 0; state <= PROJ_READ;
                    if (row != D_MODEL-1) row <= row+1;
                    else begin
                        row <= 0;
                        if (token != length-1) token <= token+1;
                        else begin
                            token <= 0;
                            if (projection < 2) projection <= projection+1;
                            else if (projection == 3) state <= FINISH;
                            else begin
                                head <= 0; key_index <= 0; component <= 0;
                                state <= SCORE_READ;
                            end
                        end
                    end
                end
                SCORE_READ: state <= SCORE_MAC;
                SCORE_MAC: begin
                    accumulator <= accumulator + product;
                    if (component == HEAD_DIM-1) state <= SCORE_SAVE;
                    else begin component <= component+1; state <= SCORE_READ; end
                end
                SCORE_SAVE: begin
                    scores[key_index] <= scaled_score;
                    if (key_index == 0 || scaled_score > max_score) max_score <= scaled_score;
                    accumulator <= 0; component <= 0;
                    // Future keys are never read: the causal mask is implicit.
                    if (key_index == token) begin
                        key_index <= 0; denominator <= 0; state <= EXP;
                    end else begin key_index <= key_index+1; state <= SCORE_READ; end
                end
                EXP: begin
                    exponentials[key_index] <= exp_lut(delta);
                    denominator <= denominator + exp_lut(delta);
                    if (key_index == token) begin key_index <= 0; state <= VALUE_READ; end
                    else key_index <= key_index+1;
                end
                VALUE_READ: state <= VALUE_MAC;
                VALUE_MAC: begin
                    accumulator <= accumulator + weighted_value;
                    if (key_index == token) state <= VALUE_SAVE;
                    else begin key_index <= key_index+1; state <= VALUE_READ; end
                end
                VALUE_SAVE: begin
                    // Normalize once per output, avoiding rounded probability vectors.
                    context_data[token*D_MODEL+head*HEAD_DIM+component] <= saturate(accumulator / denominator);
                    accumulator <= 0; key_index <= 0;
                    if (component != HEAD_DIM-1) begin component <= component+1; state <= VALUE_READ; end
                    else begin
                        component <= 0; state <= SCORE_READ;
                        if (head != NUM_HEADS-1) head <= head+1;
                        else begin
                            head <= 0;
                            if (token != length-1) token <= token+1;
                            else begin
                                projection <= 3; token <= 0; row <= 0; col <= 0; state <= PROJ_READ;
                            end
                        end
                    end
                end
                FINISH: begin done <= 1; busy <= 0; state <= IDLE; end
                default: begin state <= IDLE; busy <= 0; end
            endcase
        end
    end
endmodule
