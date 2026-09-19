`timescale 1ns/1ps

// ═══════════════════════════════════════════════════════════════
// fpGPT — Board-Level Generation Controller
//
// Wraps transformer_engine with the host-facing pieces needed to run
// autoregressive character generation over UART:
//   - byte -> token-ID mapping (matches model.CharTokenizer)
//   - prompt accumulation into the engine's token buffer
//   - a newline trigger that starts decoding
//   - signed argmax over the streamed vocabulary projection
//   - token-ID -> byte mapping on the generated token
//
// It owns the unified weight ROM (rom_sync, M10K-inferred, $readmemh).
// The engine's streamed logits are for the last sequence position, so one
// forward pass yields exactly one next token. Generated tokens are appended
// to the sequence and fed back for GEN_TOKENS steps.
// ═══════════════════════════════════════════════════════════════

module generation_controller #(
    parameter DATA_WIDTH   = 8,
    parameter ACC_WIDTH    = 32,
    parameter D_MODEL      = 64,
    parameter NUM_HEADS    = 4,
    parameter MAX_SEQ_LEN  = 64,
    parameter NUM_LAYERS   = 4,
    parameter D_FF         = 256,
    parameter VOCAB_SIZE   = 64,

    parameter Q_SHIFT      = 7,
    parameter K_SHIFT      = 7,
    parameter V_SHIFT      = 7,
    parameter OUT_SHIFT    = 7,
    parameter SCORE_MULT   = 64,
    parameter SCORE_SHIFT  = 8,

    parameter LN_SHIFT     = 7,
    parameter FC1_SHIFT    = 7,
    parameter FC2_SHIFT    = 7,
    parameter LM_SHIFT     = 7,

    parameter W_ADDR_WIDTH = 18,
    parameter ROM_DEPTH    = 212352,
    parameter ROM_MEM_FILE = "weights/weights_unified.hex",

    parameter GEN_TOKENS   = 16
) (
    input  wire        clk,
    input  wire        rst_n,

    // Host/UART byte stream
    input  wire [7:0]  token_in,
    input  wire        token_valid,
    output reg  [7:0]  token_out,
    output reg         token_out_valid,
    output reg         busy
);
    localparam TOKID_W = (VOCAB_SIZE <= 1) ? 1 : $clog2(VOCAB_SIZE);
    localparam TOK_AW  = (MAX_SEQ_LEN <= 1) ? 1 : $clog2(MAX_SEQ_LEN);
    localparam LEN_W   = ((MAX_SEQ_LEN+1) <= 1) ? 1 : $clog2(MAX_SEQ_LEN+1);
    localparam VOC_AW  = (VOCAB_SIZE <= 1) ? 1 : $clog2(VOCAB_SIZE);
    localparam signed [DATA_WIDTH-1:0] MIN_LOGIT = {1'b1, {DATA_WIDTH-1{1'b0}}};

    localparam S_COLLECT   = 2'd0,
               S_GEN_START = 2'd1,
               S_GEN_RUN   = 2'd2,
               S_GEN_EMIT  = 2'd3;

    reg [1:0]  state;
    reg [LEN_W-1:0]   seq_len;
    reg [LEN_W-1:0]   gen_count;
    reg               eng_start;
    reg               tok_load;
    reg [TOK_AW-1:0]  tok_addr;
    reg [TOKID_W-1:0] tok_data;

    reg signed [DATA_WIDTH-1:0] best_val;
    reg [VOC_AW-1:0]            best_idx;
    reg                         seen;

    wire        eng_busy, eng_done, eng_error, eng_logits_valid;
    wire [VOC_AW-1:0] eng_logits_addr;
    wire signed [DATA_WIDTH-1:0] eng_logits_data;
    wire [W_ADDR_WIDTH-1:0] eng_rom_addr;
    wire signed [DATA_WIDTH-1:0] eng_rom_data;

    // ── Engine ──
    transformer_engine #(
        .DATA_WIDTH(DATA_WIDTH), .ACC_WIDTH(ACC_WIDTH),
        .D_MODEL(D_MODEL), .NUM_HEADS(NUM_HEADS), .MAX_SEQ_LEN(MAX_SEQ_LEN),
        .NUM_LAYERS(NUM_LAYERS), .D_FF(D_FF), .VOCAB_SIZE(VOCAB_SIZE),
        .Q_SHIFT(Q_SHIFT), .K_SHIFT(K_SHIFT), .V_SHIFT(V_SHIFT),
        .OUT_SHIFT(OUT_SHIFT), .SCORE_MULT(SCORE_MULT), .SCORE_SHIFT(SCORE_SHIFT),
        .LN_SHIFT(LN_SHIFT), .FC1_SHIFT(FC1_SHIFT), .FC2_SHIFT(FC2_SHIFT),
        .LM_SHIFT(LM_SHIFT), .W_ADDR_WIDTH(W_ADDR_WIDTH)
    ) engine (
        .clk(clk), .rst_n(rst_n),
        .start(eng_start), .seq_len(seq_len),
        .tok_load(tok_load), .tok_addr(tok_addr), .tok_data(tok_data),
        .busy(eng_busy), .done(eng_done), .error(eng_error),
        .logits_valid(eng_logits_valid), .logits_addr(eng_logits_addr),
        .logits_data(eng_logits_data),
        .rom_addr(eng_rom_addr), .rom_data(eng_rom_data)
    );

    // ── Unified weight ROM ──
    rom_sync #(
        .DATA_WIDTH(DATA_WIDTH), .ADDR_WIDTH(W_ADDR_WIDTH),
        .DEPTH(ROM_DEPTH), .MEM_FILE(ROM_MEM_FILE)
    ) weight_rom_inst (
        .clk(clk), .addr(eng_rom_addr), .data(eng_rom_data)
    );

    // ── Character <-> token-ID mapping (model.CharTokenizer) ──
    // IDs 0..2 are <pad>/<sos>/<eos>; printable ASCII ' '.. maps to 3..
    function [TOKID_W-1:0] byte_to_id;
        input [7:0] b;
        begin
            if (b >= 8'd32 && b < (8'd32 + VOCAB_SIZE - 3))
                byte_to_id = b - 8'd32 + 3;
            else
                byte_to_id = {TOKID_W{1'b0}};
        end
    endfunction

    function [7:0] id_to_byte;
        input [TOKID_W-1:0] id;
        begin
            if (id >= 3)
                id_to_byte = id - 3 + 8'd32;
            else
                id_to_byte = 8'h2E;   // '.'
        end
    endfunction

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state           <= S_COLLECT;
            seq_len         <= 0;
            gen_count       <= 0;
            eng_start       <= 1'b0;
            tok_load        <= 1'b0;
            tok_addr        <= 0;
            tok_data        <= 0;
            best_val        <= MIN_LOGIT;
            best_idx        <= 0;
            seen            <= 1'b0;
            token_out       <= 0;
            token_out_valid <= 1'b0;
            busy            <= 1'b0;
        end else begin
            token_out_valid <= 1'b0;
            tok_load        <= 1'b0;

            case (state)
                // Accumulate a prompt; a CR/LF starts decoding.
                S_COLLECT: begin
                    busy      <= 1'b0;
                    eng_start <= 1'b0;
                    if (token_valid) begin
                        if (token_in == 8'h0A || token_in == 8'h0D) begin
                            if (seq_len != 0) begin
                                busy      <= 1'b1;
                                gen_count <= 0;
                                state     <= S_GEN_START;
                            end
                        end else if (seq_len < MAX_SEQ_LEN) begin
                            tok_load <= 1'b1;
                            tok_addr <= seq_len[TOK_AW-1:0];
                            tok_data <= byte_to_id(token_in);
                            seq_len  <= seq_len + 1'b1;
                        end
                    end
                end

                S_GEN_START: begin
                    eng_start <= 1'b1;
                    best_val  <= MIN_LOGIT;
                    best_idx  <= 0;
                    seen      <= 1'b0;
                    state     <= S_GEN_RUN;
                end

                S_GEN_RUN: begin
                    eng_start <= 1'b0;
                    if (eng_logits_valid) begin
                        if (!seen || eng_logits_data > best_val) begin
                            best_val <= eng_logits_data;
                            best_idx <= eng_logits_addr;
                            seen     <= 1'b1;
                        end
                    end
                    if (eng_done) state <= S_GEN_EMIT;
                end

                // Emit the winning token and feed it back into the sequence.
                S_GEN_EMIT: begin
                    token_out       <= id_to_byte(best_idx);
                    token_out_valid <= 1'b1;
                    if (seq_len < MAX_SEQ_LEN) begin
                        tok_load <= 1'b1;
                        tok_addr <= seq_len[TOK_AW-1:0];
                        tok_data <= best_idx;
                        seq_len  <= seq_len + 1'b1;
                    end
                    gen_count <= gen_count + 1'b1;
                    if (((gen_count + 1'b1) < GEN_TOKENS) && (seq_len < MAX_SEQ_LEN))
                        state <= S_GEN_START;
                    else
                        state <= S_COLLECT;
                end

                default: state <= S_COLLECT;
            endcase
        end
    end
endmodule
