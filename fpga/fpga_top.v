`timescale 1ns/1ps

// ═══════════════════════════════════════════════════════════════
// fpGPT — DE1-SoC Top Level Wrapper
//
// UART <-> generation_controller (transformer_engine + weight ROM).
// Type a prompt, send CR/LF, and the board streams generated
// characters back over UART.
//
// Model dimensions and the ROM image are parameters so a testbench can
// substitute a small model without rebuilding the board config.
// ═══════════════════════════════════════════════════════════════

module fpga_top #(
    parameter DATA_WIDTH   = 8,
    parameter D_MODEL      = 64,
    parameter NUM_HEADS    = 4,
    parameter MAX_SEQ_LEN  = 64,
    parameter NUM_LAYERS   = 4,
    parameter D_FF         = 256,
    parameter VOCAB_SIZE   = 64,
    parameter W_ADDR_WIDTH = 18,
    parameter ROM_DEPTH    = 212352,
    parameter ROM_MEM_FILE = "weights/weights_unified.hex",
    parameter GEN_TOKENS   = 16,
    parameter TOP_K        = 4,
    parameter CLK_FREQ     = 50000000,
    parameter BAUD_RATE    = 115200
) (
    // Clock & Reset
    input  wire        CLOCK_50,
    input  wire [3:0]  KEY,      // KEY[0] is active-low reset

    // UART interface
    input  wire        UART_RXD,
    output wire        UART_TXD,

    // Debug
    output wire [9:0]  LEDR
);

    wire clk   = CLOCK_50;
    wire rst_n = KEY[0];

    // ── UART RX ──
    wire [7:0] rx_data;
    wire       rx_valid;

    uart_rx #(
        .CLK_FREQ(CLK_FREQ),
        .BAUD_RATE(BAUD_RATE)
    ) rx_inst (
        .clk(clk),
        .rst_n(rst_n),
        .rx(UART_RXD),
        .data_out(rx_data),
        .valid(rx_valid)
    );

    // ── Generation controller (real transformer, not the echo stub) ──
    wire [7:0] tx_data;
    wire       tx_valid;
    wire       gen_busy;

    generation_controller #(
        .DATA_WIDTH(DATA_WIDTH),
        .D_MODEL(D_MODEL),
        .NUM_HEADS(NUM_HEADS),
        .MAX_SEQ_LEN(MAX_SEQ_LEN),
        .NUM_LAYERS(NUM_LAYERS),
        .D_FF(D_FF),
        .VOCAB_SIZE(VOCAB_SIZE),
        .W_ADDR_WIDTH(W_ADDR_WIDTH),
        .ROM_DEPTH(ROM_DEPTH),
        .ROM_MEM_FILE(ROM_MEM_FILE),
        .GEN_TOKENS(GEN_TOKENS),
        .TOP_K(TOP_K)
    ) controller_inst (
        .clk(clk),
        .rst_n(rst_n),
        .token_in(rx_data),
        .token_valid(rx_valid),
        .token_out(tx_data),
        .token_out_valid(tx_valid),
        .busy(gen_busy)
    );

    // ── UART TX ──
    wire tx_ready;

    uart_tx #(
        .CLK_FREQ(CLK_FREQ),
        .BAUD_RATE(BAUD_RATE)
    ) tx_inst (
        .clk(clk),
        .rst_n(rst_n),
        .data_in(tx_data),
        .start(tx_valid && tx_ready),
        .tx(UART_TXD),
        .ready(tx_ready)
    );

    // ── Debug LEDs ──
    // LEDR[0] = reset, LEDR[1] = UART RX active, LEDR[2] = generated token,
    // LEDR[3] = engine busy, LEDR[9:4] = last received byte[5:0].
    reg [7:0] last_rx;
    always @(posedge clk) begin
        if (rx_valid) last_rx <= rx_data;
    end

    assign LEDR[0]    = ~rst_n;
    assign LEDR[1]    = rx_valid;
    assign LEDR[2]    = tx_valid;
    assign LEDR[3]    = gen_busy;
    assign LEDR[9:4]  = last_rx[5:0];

endmodule
