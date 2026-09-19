// ═══════════════════════════════════════════════════════════════
// fpGPT — Top-Level Transformer Execution Controller
//
// Sequences a complete Micro-GPT forward pass:
//   embeddings -> LN1 -> attention -> residual ->
//   LN2 -> MLP -> residual -> final LN -> vocabulary projection
//
// The controller owns the activation buffers, all counters, the weight-ROM
// master port, and the layer loop. Sub-cores (attention, dense_layer,
// layer_norm) are driven through explicit start/done handshakes, and their
// synchronous ROM reads are time-multiplexed through one port.
//
// Weight layout is the sequential layout produced by
// compiler/ir.py:ModelIR.compute_memory_layout with biases present on the
// attention projections, LayerNorms and MLP, and no LM-head bias:
//
//   token_embedding | position_embedding | NUM_LAYERS * block
//   | final_ln.gamma | final_ln.beta | lm_head.weight
//
// where each block is:
//   ln1.g b, Q.w b, K.w b, V.w b, O.w b, ln2.g b, fc1.w b, fc2.w b
// ═══════════════════════════════════════════════════════════════

module transformer_engine #(
    parameter DATA_WIDTH      = 8,
    parameter ACC_WIDTH       = 32,
    parameter D_MODEL         = 64,
    parameter NUM_HEADS       = 4,
    parameter MAX_SEQ_LEN     = 64,
    parameter NUM_LAYERS      = 4,
    parameter D_FF            = 256,
    parameter VOCAB_SIZE      = 64,

    parameter Q_SHIFT         = 7,
    parameter K_SHIFT         = 7,
    parameter V_SHIFT         = 7,
    parameter OUT_SHIFT       = 7,
    parameter SCORE_MULT      = 64,
    parameter SCORE_SHIFT     = 8,

    parameter LN_SHIFT        = 7,
    parameter FC1_SHIFT       = 7,
    parameter FC2_SHIFT       = 7,
    parameter LM_SHIFT        = 7,

    parameter W_ADDR_WIDTH    = 32
) (
    input  wire                          clk,
    input  wire                          rst_n,

    // Host control / token load
    input  wire                          start,
    input  wire [((MAX_SEQ_LEN+1) <= 1 ? 1 : $clog2(MAX_SEQ_LEN+1))-1:0] seq_len,
    input  wire [((MAX_SEQ_LEN+1) <= 1 ? 1 : $clog2(MAX_SEQ_LEN+1))-1:0] cache_len,
    input  wire                          tok_load,
    input  wire [((MAX_SEQ_LEN <= 1) ? 1 : $clog2(MAX_SEQ_LEN))-1:0] tok_addr,
    input  wire [((VOCAB_SIZE <= 1) ? 1 : $clog2(VOCAB_SIZE))-1:0]   tok_data,

    output reg                           busy,
    output reg                           done,
    output reg                           error,

    // Streamed vocabulary projection (logits) for the last token
    output reg                           logits_valid,
    output reg  [((VOCAB_SIZE <= 1) ? 1 : $clog2(VOCAB_SIZE))-1:0] logits_addr,
    output reg  signed [DATA_WIDTH-1:0]  logits_data,

    // Unified synchronous weight ROM
    output wire [W_ADDR_WIDTH-1:0]       rom_addr,
    input  wire signed [DATA_WIDTH-1:0]  rom_data
);
    // ── Derived sizes ──
    localparam ATTN_W   = D_MODEL * D_MODEL;
    localparam BLOCK_STRIDE = 4*ATTN_W + 9*D_MODEL + 2*D_MODEL*D_FF + D_FF;
    localparam TOK_SIZE = VOCAB_SIZE * D_MODEL;
    localparam POS_SIZE = MAX_SEQ_LEN * D_MODEL;
    localparam FINAL_LN_G = TOK_SIZE + POS_SIZE + NUM_LAYERS*BLOCK_STRIDE;
    localparam FINAL_LN_B = FINAL_LN_G + D_MODEL;
    localparam LM_W       = FINAL_LN_B + D_MODEL;

    // Offsets inside one transformer block
    localparam B_LN1G = 0;
    localparam B_LN1B = B_LN1G + D_MODEL;
    localparam B_QW   = B_LN1B + D_MODEL;
    localparam B_QB   = B_QW + ATTN_W;
    localparam B_KW   = B_QB + D_MODEL;
    localparam B_KB   = B_KW + ATTN_W;
    localparam B_VW   = B_KB + D_MODEL;
    localparam B_VB   = B_VW + ATTN_W;
    localparam B_OW   = B_VB + D_MODEL;
    localparam B_OB   = B_OW + ATTN_W;
    localparam B_LN2G = B_OB + D_MODEL;
    localparam B_LN2B = B_LN2G + D_MODEL;
    localparam B_FC1W = B_LN2B + D_MODEL;
    localparam B_FC1B = B_FC1W + D_MODEL*D_FF;
    localparam B_FC2W = B_FC1B + D_FF;
    localparam B_FC2B = B_FC2W + D_FF*D_MODEL;

    localparam D_AW   = (D_MODEL   <= 1) ? 1 : $clog2(D_MODEL);
    localparam FF_AW  = (D_FF      <= 1) ? 1 : $clog2(D_FF);
    localparam VOC_AW = (VOCAB_SIZE<= 1) ? 1 : $clog2(VOCAB_SIZE);
    localparam ACT_SIZE = MAX_SEQ_LEN * D_MODEL;
    localparam ACT_AW = (ACT_SIZE  <= 1) ? 1 : $clog2(ACT_SIZE);
    localparam LEN_W  = ((MAX_SEQ_LEN+1) <= 1) ? 1 : $clog2(MAX_SEQ_LEN+1);
    localparam ATTN_W_AW = (4*ATTN_W <= 1) ? 1 : $clog2(4*ATTN_W);
    localparam ATTN_B_AW = (4*D_MODEL<= 1) ? 1 : $clog2(4*D_MODEL);
    localparam E_AW   = (ATTN_W   <= 1) ? 1 : $clog2(ATTN_W);
    localparam TOKID_W = (VOCAB_SIZE <= 1) ? 1 : $clog2(VOCAB_SIZE);

    // ── FSM ──
    localparam S_IDLE         = 5'd0,
               S_EMB_TOK_ISS  = 5'd1,  S_EMB_TOK_WAIT = 5'd2,
               S_EMB_POS_ISS  = 5'd3,  S_EMB_POS_WAIT = 5'd4,
               S_LN1_START    = 5'd5,  S_LN1_RUN      = 5'd6,
               S_AW_ISS       = 5'd7,  S_AW_WAIT      = 5'd8,
               S_AB_ISS       = 5'd9,  S_AB_WAIT      = 5'd10,
               S_ATTN_X       = 5'd11, S_ATTN_START   = 5'd12, S_ATTN_RUN = 5'd13,
               S_LN2_START    = 5'd14, S_LN2_RUN      = 5'd15,
               S_FC1_START    = 5'd16, S_FC1_RUN      = 5'd17,
               S_FC2_START    = 5'd18, S_FC2_RUN      = 5'd19,
               S_FLN_START    = 5'd20, S_FLN_RUN      = 5'd21,
               S_LM_START     = 5'd22, S_LM_RUN       = 5'd23,
               S_FINISH       = 5'd24;

    localparam OWN_CTRL = 3'd0, OWN_LN = 3'd1, OWN_FC1 = 3'd2,
               OWN_FC2 = 3'd3, OWN_LM = 3'd4;

    reg [4:0] state;
    reg [2:0] rom_owner;

    reg [W_ADDR_WIDTH-1:0] blk_base;
    reg [W_ADDR_WIDTH-1:0] ln_g_base, ln_b_base;
    reg [W_ADDR_WIDTH-1:0] fc1_w_base, fc1_b_base, fc2_w_base, fc2_b_base, lm_w_base;
    reg [ACT_AW-1:0]       x_base;
    reg [ACT_AW-1:0]       ln_tok;
    reg [LEN_W-1:0]        seq_len_reg;

    integer l_cnt;
    reg [ACT_AW-1:0] t_cnt;
    reg [D_AW-1:0]   c_cnt;
    reg [E_AW-1:0]   e_cnt;
    reg [2:0]        p_cnt;
    reg signed [DATA_WIDTH-1:0] tok_val;

    // Activation storage
    reg signed [DATA_WIDTH-1:0] act [0:ACT_SIZE-1];       // hidden state
    reg signed [DATA_WIDTH-1:0] tmp [0:ACT_SIZE-1];       // LN output / attention input
    reg signed [DATA_WIDTH-1:0] hidden [0:D_FF-1];        // MLP intermediate
    reg [TOKID_W-1:0] tokens [0:MAX_SEQ_LEN-1];

    // ── Sub-core interfaces ──
    reg                    attn_x_load;
    wire [ACT_AW-1:0]      attn_x_addr = t_cnt*D_MODEL + c_cnt;
    reg                    attn_w_load;
    reg  [ATTN_W_AW-1:0]   attn_w_addr;
    reg                    attn_b_load;
    reg  [ATTN_B_AW-1:0]   attn_b_addr;
    reg                    attn_start;
    reg  [LEN_W-1:0]       attn_seq_len;
    reg  [LEN_W-1:0]       attn_cache_len;
    wire                   attn_busy, attn_done, attn_error, attn_y_valid;
    wire [ACT_AW-1:0]      attn_y_addr;
    wire signed [DATA_WIDTH-1:0] attn_y_data;
    wire signed [63:0]     attn_b_data = {{(64-DATA_WIDTH){rom_data[DATA_WIDTH-1]}}, rom_data};
    wire signed [DATA_WIDTH-1:0] attn_w_data = rom_data;

    reg                    ln_start;
    wire [D_AW-1:0]        ln_x_addr;
    wire [W_ADDR_WIDTH-1:0] ln_rom_addr;
    wire                   ln_y_wr_en;
    wire [D_AW-1:0]        ln_y_addr;
    wire signed [DATA_WIDTH-1:0] ln_y_data;
    wire                   ln_done;

    reg                    fc1_start;
    wire [W_ADDR_WIDTH-1:0] fc1_w_addr;
    wire [D_AW-1:0]        fc1_x_addr;
    wire                   fc1_y_wr_en;
    wire [FF_AW-1:0]       fc1_y_addr;
    wire signed [DATA_WIDTH-1:0] fc1_y_data;
    wire                   fc1_done;

    reg                    fc2_start;
    wire [W_ADDR_WIDTH-1:0] fc2_w_addr;
    wire [FF_AW-1:0]       fc2_x_addr;
    wire                   fc2_y_wr_en;
    wire [D_AW-1:0]        fc2_y_addr;
    wire signed [DATA_WIDTH-1:0] fc2_y_data;
    wire                   fc2_done;

    reg                    lm_start;
    wire [W_ADDR_WIDTH-1:0] lm_w_addr;
    wire [D_AW-1:0]        lm_x_addr;
    wire                   lm_y_wr_en;
    wire [VOC_AW-1:0]      lm_y_addr;
    wire signed [DATA_WIDTH-1:0] lm_y_data;
    wire                   lm_done;

    wire signed [DATA_WIDTH-1:0] fc1_x_data = tmp[x_base + fc1_x_addr];
    wire signed [DATA_WIDTH-1:0] fc2_x_data = hidden[fc2_x_addr];
    wire signed [DATA_WIDTH-1:0] lm_x_data  = tmp[x_base + lm_x_addr];
    wire signed [DATA_WIDTH-1:0] ln_x_data  = act[ln_tok*D_MODEL + ln_x_addr];
    wire signed [DATA_WIDTH-1:0] attn_x_data = tmp[attn_x_addr];

    // Controller-driven ROM address for embedding and attention weight load
    wire [W_ADDR_WIDTH-1:0] ctrl_rom_addr;
    assign ctrl_rom_addr =
        (state == S_EMB_TOK_ISS) ? (tokens[t_cnt]*D_MODEL + c_cnt)
      : (state == S_EMB_POS_ISS) ? (TOK_SIZE + t_cnt*D_MODEL + c_cnt)
      : (state == S_AW_ISS)      ? (blk_base + B_QW + p_cnt*(ATTN_W + D_MODEL) + e_cnt)
      : (state == S_AB_ISS)      ? (blk_base + B_QW + p_cnt*(ATTN_W + D_MODEL) + ATTN_W + e_cnt)
      : {W_ADDR_WIDTH{1'b0}};

    assign rom_addr = (rom_owner == OWN_LN)  ? ln_rom_addr
                    : (rom_owner == OWN_FC1) ? fc1_w_addr
                    : (rom_owner == OWN_FC2) ? fc2_w_addr
                    : (rom_owner == OWN_LM)  ? lm_w_addr
                    : ctrl_rom_addr;

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

    // ── Sub-core instances ──
    attention #(
        .DATA_WIDTH(DATA_WIDTH), .D_MODEL(D_MODEL), .NUM_HEADS(NUM_HEADS),
        .MAX_SEQ_LEN(MAX_SEQ_LEN), .NUM_LAYERS(NUM_LAYERS), .Q_SHIFT(Q_SHIFT), .K_SHIFT(K_SHIFT),
        .V_SHIFT(V_SHIFT), .OUT_SHIFT(OUT_SHIFT),
        .SCORE_MULT(SCORE_MULT), .SCORE_SHIFT(SCORE_SHIFT)
    ) attn_inst (
        .clk(clk), .rst_n(rst_n),
        .x_load(attn_x_load), .x_load_addr(attn_x_addr), .x_load_data(attn_x_data),
        .w_load(attn_w_load), .w_load_addr(attn_w_addr), .w_load_data(attn_w_data),
        .b_load(attn_b_load), .b_load_addr(attn_b_addr), .b_load_data(attn_b_data),
        .start(attn_start), .seq_len(attn_seq_len), .cache_len(attn_cache_len),
        .layer_idx(l_cnt[(NUM_LAYERS<=1 ? 1 : $clog2(NUM_LAYERS))-1:0]),
        .busy(attn_busy), .done(attn_done), .error(attn_error),
        .y_valid(attn_y_valid), .y_addr(attn_y_addr), .y_data(attn_y_data)
    );

    layer_norm #(
        .DATA_WIDTH(DATA_WIDTH), .ACC_WIDTH(ACC_WIDTH), .N(D_MODEL),
        .SHIFT(LN_SHIFT), .ROM_ADDR_WIDTH(W_ADDR_WIDTH)
    ) ln_inst (
        .clk(clk), .rst_n(rst_n), .start(ln_start),
        .x_addr(ln_x_addr), .x_data(ln_x_data),
        .rom_addr(ln_rom_addr), .rom_data(rom_data),
        .g_base(ln_g_base), .b_base(ln_b_base),
        .y_wr_en(ln_y_wr_en), .y_addr(ln_y_addr), .y_data(ln_y_data), .done(ln_done)
    );

    dense_layer #(
        .DATA_WIDTH(DATA_WIDTH), .ACC_WIDTH(ACC_WIDTH),
        .IN_FEATURES(D_MODEL), .OUT_FEATURES(D_FF), .SHIFT_BITS(FC1_SHIFT),
        .ACTIVATION("gelu"), .W_ADDR_WIDTH(W_ADDR_WIDTH), .HAS_BIAS(1)
    ) fc1_inst (
        .clk(clk), .rst_n(rst_n), .start(fc1_start), .done(fc1_done),
        .w_base(fc1_w_base), .b_base(fc1_b_base),
        .w_addr(fc1_w_addr), .w_data(rom_data),
        .x_addr(fc1_x_addr), .x_data(fc1_x_data),
        .y_wr_en(fc1_y_wr_en), .y_addr(fc1_y_addr), .y_data(fc1_y_data)
    );

    dense_layer #(
        .DATA_WIDTH(DATA_WIDTH), .ACC_WIDTH(ACC_WIDTH),
        .IN_FEATURES(D_FF), .OUT_FEATURES(D_MODEL), .SHIFT_BITS(FC2_SHIFT),
        .ACTIVATION("none"), .W_ADDR_WIDTH(W_ADDR_WIDTH), .HAS_BIAS(1)
    ) fc2_inst (
        .clk(clk), .rst_n(rst_n), .start(fc2_start), .done(fc2_done),
        .w_base(fc2_w_base), .b_base(fc2_b_base),
        .w_addr(fc2_w_addr), .w_data(rom_data),
        .x_addr(fc2_x_addr), .x_data(fc2_x_data),
        .y_wr_en(fc2_y_wr_en), .y_addr(fc2_y_addr), .y_data(fc2_y_data)
    );

    dense_layer #(
        .DATA_WIDTH(DATA_WIDTH), .ACC_WIDTH(ACC_WIDTH),
        .IN_FEATURES(D_MODEL), .OUT_FEATURES(VOCAB_SIZE), .SHIFT_BITS(LM_SHIFT),
        .ACTIVATION("none"), .W_ADDR_WIDTH(W_ADDR_WIDTH), .HAS_BIAS(0)
    ) lm_inst (
        .clk(clk), .rst_n(rst_n), .start(lm_start), .done(lm_done),
        .w_base(lm_w_base), .b_base({W_ADDR_WIDTH{1'b0}}),
        .w_addr(lm_w_addr), .w_data(rom_data),
        .x_addr(lm_x_addr), .x_data(lm_x_data),
        .y_wr_en(lm_y_wr_en), .y_addr(lm_y_addr), .y_data(lm_y_data)
    );

    // ── Token load (host) ──
    integer ti;
    always @(posedge clk) begin
        if (tok_load && !busy && tok_addr < MAX_SEQ_LEN)
            tokens[tok_addr] <= tok_data;
    end

    // ── Main sequencing FSM ──
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE;
            busy <= 1'b0; done <= 1'b0; error <= 1'b0;
            logits_valid <= 1'b0; logits_addr <= 0; logits_data <= 0;
            rom_owner <= OWN_CTRL;
            blk_base <= 0;
            ln_g_base <= 0; ln_b_base <= 0;
            fc1_w_base <= 0; fc1_b_base <= 0; fc2_w_base <= 0; fc2_b_base <= 0; lm_w_base <= 0;
            x_base <= 0; ln_tok <= 0; seq_len_reg <= 0;
            l_cnt <= 0; t_cnt <= 0; c_cnt <= 0; e_cnt <= 0; p_cnt <= 0;
            tok_val <= 0;
            attn_x_load <= 0; attn_w_load <= 0; attn_w_addr <= 0;
            attn_b_load <= 0; attn_b_addr <= 0; attn_start <= 0; attn_seq_len <= 0; attn_cache_len <= 0;
            ln_start <= 0;
            fc1_start <= 0; fc2_start <= 0; lm_start <= 0;
        end else begin
            done <= 1'b0;
            logits_valid <= 1'b0;

            case (state)
                S_IDLE: begin
                    if (start) begin
                        if (seq_len == 0 || seq_len > MAX_SEQ_LEN) begin
                            error <= 1'b1; done <= 1'b1;
                        end else begin
                            error <= 1'b0; busy <= 1'b1;
                            seq_len_reg <= seq_len;
                            t_cnt <= cache_len; c_cnt <= 0; e_cnt <= 0; p_cnt <= 0;
                            l_cnt <= 0; blk_base <= TOK_SIZE + POS_SIZE;
                            rom_owner <= OWN_CTRL;
                            state <= S_EMB_TOK_ISS;
                        end
                    end
                end

                // ── Token + position embedding ──
                S_EMB_TOK_ISS:  state <= S_EMB_TOK_WAIT;
                S_EMB_TOK_WAIT: begin tok_val <= rom_data; state <= S_EMB_POS_ISS; end
                S_EMB_POS_ISS:  state <= S_EMB_POS_WAIT;
                S_EMB_POS_WAIT: begin
                    act[t_cnt*D_MODEL + c_cnt] <= saturate(tok_val + rom_data);
                    if (c_cnt == D_MODEL-1) begin
                        c_cnt <= 0;
                        if (t_cnt == seq_len_reg-1) begin
                            t_cnt <= cache_len;
                            state <= S_LN1_START;
                        end else begin
                            t_cnt <= t_cnt + 1;
                            state <= S_EMB_TOK_ISS;
                        end
                    end else begin
                        c_cnt <= c_cnt + 1;
                        state <= S_EMB_TOK_ISS;
                    end
                end

                // ── Pre-attention LayerNorm (per token) ──
                S_LN1_START: begin
                    ln_g_base <= blk_base + B_LN1G;
                    ln_b_base <= blk_base + B_LN1B;
                    ln_tok <= t_cnt;
                    ln_start <= 1'b1;
                    rom_owner <= OWN_LN;
                    state <= S_LN1_RUN;
                end
                S_LN1_RUN: begin
                    ln_start <= 1'b0;
                    if (ln_y_wr_en) tmp[ln_tok*D_MODEL + ln_y_addr] <= ln_y_data;
                    if (ln_done) begin
                        if (t_cnt == seq_len_reg-1) begin
                            t_cnt <= cache_len; p_cnt <= 0; e_cnt <= 0;
                            rom_owner <= OWN_CTRL;
                            state <= S_AW_ISS;
                        end else begin
                            t_cnt <= t_cnt + 1;
                            state <= S_LN1_START;
                        end
                    end
                end

                // ── Load attention weights/biases from ROM ──
                S_AW_ISS: begin
                    attn_w_load <= 1'b1;
                    attn_w_addr <= p_cnt*ATTN_W + e_cnt;
                    state <= S_AW_WAIT;
                end
                S_AW_WAIT: begin
                    attn_w_load <= 1'b0;
                    if (e_cnt == ATTN_W-1) begin e_cnt <= 0; state <= S_AB_ISS; end
                    else begin e_cnt <= e_cnt + 1; state <= S_AW_ISS; end
                end
                S_AB_ISS: begin
                    attn_b_load <= 1'b1;
                    attn_b_addr <= p_cnt*D_MODEL + e_cnt;
                    state <= S_AB_WAIT;
                end
                S_AB_WAIT: begin
                    attn_b_load <= 1'b0;
                    if (e_cnt == D_MODEL-1) begin
                        e_cnt <= 0;
                        if (p_cnt == 3) begin
                            t_cnt <= cache_len; c_cnt <= 0;
                            attn_x_load <= 1'b1;
                            state <= S_ATTN_X;
                        end else begin
                            p_cnt <= p_cnt + 1;
                            state <= S_AW_ISS;
                        end
                    end else begin
                        e_cnt <= e_cnt + 1;
                        state <= S_AB_ISS;
                    end
                end

                // ── Load attention input from LN1 output ──
                S_ATTN_X: begin
                    if (c_cnt == D_MODEL-1) begin
                        c_cnt <= 0;
                        if (t_cnt == seq_len_reg-1) begin
                            t_cnt <= cache_len;
                            attn_x_load <= 1'b0;
                            state <= S_ATTN_START;
                        end else begin
                            t_cnt <= t_cnt + 1;
                        end
                    end else begin
                        c_cnt <= c_cnt + 1;
                    end
                end

                // ── Run attention, fold output into residual stream ──
                S_ATTN_START: begin
                    attn_start <= 1'b1;
                    attn_seq_len <= seq_len_reg;
                    attn_cache_len <= cache_len;
                    state <= S_ATTN_RUN;
                end
                S_ATTN_RUN: begin
                    attn_start <= 1'b0;
                    if (attn_y_valid)
                        act[attn_y_addr] <= saturate(act[attn_y_addr] + attn_y_data);
                    if (attn_done) begin
                        t_cnt <= cache_len;
                        state <= S_LN2_START;
                    end
                end

                // ── Pre-MLP LayerNorm ──
                S_LN2_START: begin
                    ln_g_base <= blk_base + B_LN2G;
                    ln_b_base <= blk_base + B_LN2B;
                    ln_tok <= t_cnt;
                    ln_start <= 1'b1;
                    rom_owner <= OWN_LN;
                    state <= S_LN2_RUN;
                end
                S_LN2_RUN: begin
                    ln_start <= 1'b0;
                    if (ln_y_wr_en) tmp[ln_tok*D_MODEL + ln_y_addr] <= ln_y_data;
                    if (ln_done) begin
                        if (t_cnt == seq_len_reg-1) begin
                            t_cnt <= cache_len;
                            state <= S_FC1_START;
                        end else begin
                            t_cnt <= t_cnt + 1;
                            state <= S_LN2_START;
                        end
                    end
                end

                // ── MLP: fc1 (gelu) then fc2, then residual ──
                S_FC1_START: begin
                    fc1_start <= 1'b1;
                    fc1_w_base <= blk_base + B_FC1W;
                    fc1_b_base <= blk_base + B_FC1B;
                    x_base <= t_cnt * D_MODEL;
                    rom_owner <= OWN_FC1;
                    state <= S_FC1_RUN;
                end
                S_FC1_RUN: begin
                    fc1_start <= 1'b0;
                    if (fc1_y_wr_en) hidden[fc1_y_addr] <= fc1_y_data;
                    if (fc1_done) state <= S_FC2_START;
                end
                S_FC2_START: begin
                    fc2_start <= 1'b1;
                    fc2_w_base <= blk_base + B_FC2W;
                    fc2_b_base <= blk_base + B_FC2B;
                    rom_owner <= OWN_FC2;
                    state <= S_FC2_RUN;
                end
                S_FC2_RUN: begin
                    fc2_start <= 1'b0;
                    if (fc2_y_wr_en)
                        act[t_cnt*D_MODEL + fc2_y_addr] <=
                            saturate(act[t_cnt*D_MODEL + fc2_y_addr] + fc2_y_data);
                    if (fc2_done) begin
                        if (t_cnt == seq_len_reg-1) begin
                            t_cnt <= cache_len;
                            if (l_cnt == NUM_LAYERS-1) begin
                                state <= S_FLN_START;
                            end else begin
                                l_cnt <= l_cnt + 1;
                                blk_base <= blk_base + BLOCK_STRIDE;
                                state <= S_LN1_START;
                            end
                        end else begin
                            t_cnt <= t_cnt + 1;
                            state <= S_FC1_START;
                        end
                    end
                end

                // ── Final LayerNorm ──
                S_FLN_START: begin
                    ln_g_base <= FINAL_LN_G;
                    ln_b_base <= FINAL_LN_B;
                    ln_tok <= t_cnt;
                    ln_start <= 1'b1;
                    rom_owner <= OWN_LN;
                    state <= S_FLN_RUN;
                end
                S_FLN_RUN: begin
                    ln_start <= 1'b0;
                    if (ln_y_wr_en) tmp[ln_tok*D_MODEL + ln_y_addr] <= ln_y_data;
                    if (ln_done) begin
                        if (t_cnt == seq_len_reg-1) begin
                            t_cnt <= cache_len;
                            state <= S_LM_START;
                        end else begin
                            t_cnt <= t_cnt + 1;
                            state <= S_FLN_START;
                        end
                    end
                end

                // ── Vocabulary projection of the last token ──
                S_LM_START: begin
                    lm_start <= 1'b1;
                    lm_w_base <= LM_W;
                    x_base <= (seq_len_reg-1) * D_MODEL;
                    rom_owner <= OWN_LM;
                    state <= S_LM_RUN;
                end
                S_LM_RUN: begin
                    lm_start <= 1'b0;
                    if (lm_y_wr_en) begin
                        logits_valid <= 1'b1;
                        logits_addr  <= lm_y_addr;
                        logits_data  <= lm_y_data;
                    end
                    if (lm_done) state <= S_FINISH;
                end

                S_FINISH: begin
                    done <= 1'b1;
                    busy <= 1'b0;
                    state <= S_IDLE;
                end

                default: state <= S_IDLE;
            endcase
        end
    end
endmodule
