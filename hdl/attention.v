// Sequential causal multi-head self-attention, including Q/K/V/out projections.
// See hdl/attention.md for the loading protocol and fixed-point contract.
//
// All matrix storage is written as simple dual-port block RAM so Quartus can
// infer M10K: one registered write port and one registered read port per
// memory. The FSM already consumes operands one cycle after issuing the read
// address, so the RAM's registered output matches the original timing.
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
    parameter LEN_WIDTH = $clog2(MAX_SEQ_LEN+1),
    parameter NUM_LAYERS = 4
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
    input wire [LEN_WIDTH-1:0] cache_len,
    input wire [(NUM_LAYERS<=1 ? 1 : $clog2(NUM_LAYERS))-1:0] layer_idx,
    output reg busy, done, error,
    output reg y_valid,
    output reg [X_ADDR_WIDTH-1:0] y_addr,
    output reg signed [DATA_WIDTH-1:0] y_data
);
    localparam HEAD_DIM = D_MODEL / NUM_HEADS;
    localparam KV_DEPTH = NUM_LAYERS * MAX_SEQ_LEN * D_MODEL;
    localparam KV_AW = (KV_DEPTH <= 1) ? 1 : $clog2(KV_DEPTH);
    localparam SC_AW = (MAX_SEQ_LEN <= 1) ? 1 : $clog2(MAX_SEQ_LEN);
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
               EXP_READ=7, EXP=8, VALUE_READ=9, VALUE_MAC=10, VALUE_SAVE=11,
               VALUE_DIV=12, FINISH=13;
    reg [3:0] state;

    // ── Matrix storage: simple dual-port, M10K-inferable ──
    (* ramstyle = "M10K" *) reg signed [DATA_WIDTH-1:0] inputs [0:MAX_SEQ_LEN*D_MODEL-1];
    (* ramstyle = "M10K" *) reg signed [DATA_WIDTH-1:0] weights [0:4*D_MODEL*D_MODEL-1];
    reg signed [63:0] biases [0:4*D_MODEL-1];
    (* ramstyle = "M10K" *) reg signed [DATA_WIDTH-1:0] queries [0:MAX_SEQ_LEN*D_MODEL-1];
    (* ramstyle = "M10K" *) reg signed [DATA_WIDTH-1:0] k_cache [0:KV_DEPTH-1];
    (* ramstyle = "M10K" *) reg signed [DATA_WIDTH-1:0] v_cache [0:KV_DEPTH-1];
    (* ramstyle = "M10K" *) reg signed [DATA_WIDTH-1:0] context_data [0:MAX_SEQ_LEN*D_MODEL-1];
    reg signed [63:0] scores [0:MAX_SEQ_LEN-1];
    reg [15:0] exponentials [0:MAX_SEQ_LEN-1];

    // Registered read data
    reg signed [DATA_WIDTH-1:0] inputs_q, weights_q, queries_q, kcache_q,
                                vcache_q, context_q;
    reg signed [63:0] bias_q, scores_q;
    reg [15:0] exp_operand;

    integer length, projection, token, row, col, head, key_index, component;
    reg signed [63:0] accumulator, max_score;
    reg signed [63:0] denominator;
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

    // ── Read addresses (presented one cycle before the data is consumed) ──
    wire [X_ADDR_WIDTH-1:0] input_raddr   = token*D_MODEL + col;
    wire [W_ADDR_WIDTH-1:0] weight_raddr  = projection*D_MODEL*D_MODEL + row*D_MODEL + col;
    wire [X_ADDR_WIDTH-1:0] query_raddr   = token*D_MODEL + head*HEAD_DIM + component;
    wire [KV_AW-1:0]        kcache_raddr  = layer_idx*MAX_SEQ_LEN*D_MODEL + key_index*D_MODEL + head*HEAD_DIM + component;
    wire [KV_AW-1:0]        vcache_raddr  = kcache_raddr;
    wire [X_ADDR_WIDTH-1:0] context_raddr = token*D_MODEL + col;
    wire [B_ADDR_WIDTH-1:0] bias_raddr    = projection*D_MODEL + row;
    wire [SC_AW-1:0]        scores_raddr  = (state == EXP && key_index < token) ? (key_index + 1)
                                          : key_index;

    // ── Write ports (one write + one registered read per memory) ──
    wire load_en = !busy && rst_n;
    wire inputs_we  = load_en && x_load && (x_load_addr < MAX_SEQ_LEN*D_MODEL);
    wire weights_we = load_en && w_load && (w_load_addr < 4*D_MODEL*D_MODEL);
    wire biases_we  = load_en && b_load && (b_load_addr < 4*D_MODEL);

    // ── Sequential divider for the value normalization (accumulator/denominator) ──
    reg              div_start;
    reg signed [63:0] div_num, div_den;
    reg              div_busy;
    reg [64:0]       div_rem;
    reg [63:0]       div_q, div_ua, div_ub;
    reg              div_neg;
    reg [6:0]        div_cnt;
    wire signed [63:0] div_quot = div_neg ? -div_q : div_q;
    wire             div_done = div_busy && (div_cnt == 0);
    wire [64:0]      div_shift = {div_rem[63:0], div_ua[63]};
    wire [64:0]      div_ubx   = {1'b0, div_ub};
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
            div_cnt  <= 64;
            div_busy <= 1'b1;
        end else if (div_busy) begin
            if (div_cnt != 0) begin
                div_rem <= div_ge ? (div_shift - div_ubx) : div_shift;
                div_q   <= div_ge ? {div_q[62:0], 1'b1} : {div_q[62:0], 1'b0};
                div_ua  <= {div_ua[62:0], 1'b0};
                div_cnt <= div_cnt - 1'b1;
            end else begin
                div_busy <= 1'b0;
            end
        end
    end

    wire signed [63:0] proj_sum  = accumulator + bias_q;
    wire signed [DATA_WIDTH-1:0] proj_wdata = saturate(requant(proj_sum, projection_shift));
    wire proj_save = (state == PROJ_SAVE);
    wire queries_we = proj_save && (projection == 0);
    wire kcache_we  = proj_save && (projection == 1);
    wire vcache_we  = proj_save && (projection == 2);
    wire context_we = (state == VALUE_DIV) && div_done;
    wire signed [DATA_WIDTH-1:0] context_wdata = saturate(div_quot);

    wire signed [63:0] scaled_score = (accumulator * SCORE_MULT) >>> SCORE_SHIFT;
    wire signed [63:0] delta = max_score - scores_q;
    wire signed [DATA_WIDTH+16:0] weighted_value = vcache_q * $signed({1'b0, exp_operand});

    always @(posedge clk) begin
        if (inputs_we) inputs[x_load_addr] <= x_load_data;
        inputs_q <= inputs[input_raddr];
    end
    always @(posedge clk) begin
        if (weights_we) weights[w_load_addr] <= w_load_data;
        weights_q <= weights[weight_raddr];
    end
    always @(posedge clk) begin
        if (biases_we) biases[b_load_addr] <= b_load_data;
        bias_q <= biases[bias_raddr];
    end
    always @(posedge clk) begin
        if (queries_we) queries[token*D_MODEL+row] <= proj_wdata;
        queries_q <= queries[query_raddr];
    end
    always @(posedge clk) begin
        if (kcache_we) k_cache[layer_idx*MAX_SEQ_LEN*D_MODEL + token*D_MODEL+row] <= proj_wdata;
        kcache_q <= k_cache[kcache_raddr];
    end
    always @(posedge clk) begin
        if (vcache_we) v_cache[layer_idx*MAX_SEQ_LEN*D_MODEL + token*D_MODEL+row] <= proj_wdata;
        vcache_q <= v_cache[vcache_raddr];
    end
    always @(posedge clk) begin
        if (context_we) context_data[token*D_MODEL+head*HEAD_DIM+component] <= context_wdata;
        context_q <= context_data[context_raddr];
    end
    always @(posedge clk) begin
        if (state == SCORE_SAVE) scores[key_index] <= scaled_score;
        scores_q <= scores[scores_raddr];
    end
    always @(posedge clk) begin
        if (state == EXP) exponentials[key_index] <= exp_lut(delta);
        exp_operand <= exponentials[key_index];
    end

    // ── Operand selection for the MAC states ──
    wire signed [DATA_WIDTH-1:0] proj_a = (projection == 3) ? context_q : inputs_q;
    wire signed [DATA_WIDTH-1:0] proj_b = weights_q;
    wire signed [DATA_WIDTH-1:0] score_a = queries_q;
    wire signed [DATA_WIDTH-1:0] score_b = kcache_q;
    reg signed [2*DATA_WIDTH-1:0] product;
    always @(*) begin
        case (state)
            PROJ_MAC:  product = proj_a * proj_b;
            SCORE_MAC: product = score_a * score_b;
            default:   product = {2*DATA_WIDTH{1'b0}};
        endcase
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE; busy <= 0; done <= 0; error <= 0;
            y_valid <= 0; y_addr <= 0; y_data <= 0;
            length <= 0; projection <= 0; token <= 0; row <= 0; col <= 0;
            head <= 0; key_index <= 0; component <= 0;
            accumulator <= 0; max_score <= 0; denominator <= 0;
            div_start <= 1'b0; div_num <= 0; div_den <= 0;
        end else begin
            done <= 0; y_valid <= 0;
            case (state)
                IDLE: if (start) begin
                    error <= 0;
                    if (seq_len == 0 || seq_len > MAX_SEQ_LEN || cache_len >= seq_len) begin
                        error <= 1; done <= 1;
                    end else begin
                        length <= seq_len; busy <= 1; projection <= 0;
                        token <= cache_len; row <= 0; col <= 0;
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
                        0: ; // queries write handled by the memory port
                        1: ; // k_cache write handled by the memory port
                        2: ; // v_cache write handled by the memory port
                        3: begin
                            y_data <= proj_wdata;
                            y_addr <= token*D_MODEL+row; y_valid <= 1;
                        end
                    endcase
                    accumulator <= 0; col <= 0; state <= PROJ_READ;
                    if (row != D_MODEL-1) row <= row+1;
                    else begin
                        row <= 0;
                        if (token != length-1) token <= token+1;
                        else begin
                            token <= cache_len;
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
                    if (key_index == 0 || scaled_score > max_score) max_score <= scaled_score;
                    accumulator <= 0; component <= 0;
                    // Future keys are never read: the causal mask is implicit.
                    if (key_index == token) begin
                        key_index <= 0; denominator <= 0; state <= EXP_READ;
                    end else begin key_index <= key_index+1; state <= SCORE_READ; end
                end
                EXP_READ: state <= EXP;
                EXP: begin
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
                    div_num <= accumulator;
                    div_den <= denominator;
                    div_start <= 1'b1;
                    state <= VALUE_DIV;
                end
                VALUE_DIV: begin
                    div_start <= 1'b0;
                    if (div_done) begin
                        accumulator <= 0; key_index <= 0;
                        if (component != HEAD_DIM-1) begin component <= component+1; state <= VALUE_READ; end
                        else begin
                            component <= 0; state <= SCORE_READ;
                            if (head != NUM_HEADS-1) head <= head+1;
                            else begin
                                head <= 0;
                                if (token != length-1) token <= token+1;
                                else begin
                                    projection <= 3; token <= cache_len; row <= 0; col <= 0; state <= PROJ_READ;
                                end
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
