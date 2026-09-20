`timescale 1ns/1ps

// ═══════════════════════════════════════════════════════════════
// fpGPT — DE1-SoC Top Level Wrapper
//
// UART <-> generation_controller (transformer_engine + weight ROM).
// Type a prompt, send CR/LF, and the board streams generated
// characters back over UART.
//
// Clocking:
//   CLOCK_50 (PIN_AF14) -> pll_150 -> clk_sys (150 MHz) on hardware.
//   In simulation (FPGPT_USE_PLL undefined) clk_sys follows CLOCK_50 so the
//   testbenches keep their 50 MHz timing; pass CLK_FREQ=50000000 there.
//
// Model geometry and the engine shifts come from board_params.vh (stream 1:
// compile.py -> build/rtl/board_params.vh, copied here by build.tcl). It is
// included at file scope because the FPGPT_* macros also seed the module
// parameter defaults. Define FPGPT_TINY_TEST to substitute a small model for
// the smoke testbench.
// ═══════════════════════════════════════════════════════════════

`ifdef FPGPT_TINY_TEST
    // Small model for tests/tb_fpga_top.v; mirrors board_params.vh.
    `define FPGPT_BOARD_PARAMS_VH
    `define FPGPT_DATA_WIDTH     8
    `define FPGPT_D_MODEL        4
    `define FPGPT_NUM_HEADS      2
    `define FPGPT_MAX_SEQ_LEN    4
    `define FPGPT_NUM_LAYERS     1
    `define FPGPT_D_FF           4
    `define FPGPT_VOCAB_SIZE     8
    `define FPGPT_W_ADDR_WIDTH   12
    `define FPGPT_ROM_DEPTH      4096
    `define FPGPT_ROM_MEM_FILE   "no_such_file.hex"
    `define FPGPT_GEN_TOKENS     1
    `define FPGPT_NUM_PES        1
    `define FPGPT_Q_SHIFT        0
    `define FPGPT_K_SHIFT        0
    `define FPGPT_V_SHIFT        0
    `define FPGPT_OUT_SHIFT      0
    `define FPGPT_SCORE_MULT     64
    `define FPGPT_SCORE_SHIFT    8
    `define FPGPT_LN_SHIFT       7
    `define FPGPT_FC1_SHIFT      0
    `define FPGPT_FC2_SHIFT      0
    `define FPGPT_LM_SHIFT       0
    `define FPGPT_SYS_CLK_HZ     50000000
    `define FPGPT_BAUD_RATE      115200
`else
    `include "board_params.vh"
`endif

module fpga_top #(
    parameter ROM_MEM_FILE = `FPGPT_ROM_MEM_FILE,
    parameter GEN_TOKENS   = `FPGPT_GEN_TOKENS,
    parameter NUM_PES      = `FPGPT_NUM_PES,
    parameter TOP_K        = 4,
    parameter CLK_FREQ     = `FPGPT_SYS_CLK_HZ,
    parameter BAUD_RATE    = `FPGPT_BAUD_RATE,
    parameter LN_SHIFT     = `FPGPT_LN_SHIFT
) (
    // Clock & Reset
    input  wire        CLOCK_50, // 50 MHz oscillator, PIN_AF14
    input  wire [3:0]  KEY,      // KEY[0] is active-low reset

    // UART interface (external USB-TTL adapter on GPIO_0[0:1])
    input  wire        UART_RXD, // GPIO_0[0] = PIN_AC18
    output wire        UART_TXD, // GPIO_0[1] = PIN_Y17

    // Debug
    output wire [9:0]  LEDR
);

    wire rst_n = KEY[0];

    // ── System clock: 150 MHz on hardware, passthrough in simulation ──
`ifdef FPGPT_USE_PLL
    wire clk_sys;
    wire pll_locked;
    pll_150 pll_inst (
        .refclk   (CLOCK_50),
        .rst      (~rst_n),
        .outclk_0 (clk_sys),
        .locked   (pll_locked)
    );
`else
    wire clk_sys    = CLOCK_50;
    wire pll_locked = rst_n;
`endif

    // ── UART RX ──
    wire [7:0] rx_data;
    wire       rx_valid;

    uart_rx #(
        .CLK_FREQ(CLK_FREQ),
        .BAUD_RATE(BAUD_RATE)
    ) rx_inst (
        .clk(clk_sys),
        .rst_n(rst_n),
        .rx(UART_RXD),
        .data_out(rx_data),
        .valid(rx_valid)
    );

    // ── Generation controller (real transformer, not the echo stub) ──
    wire [7:0]  tx_data;
    wire        tx_valid;
    wire        gen_busy;
    wire [31:0] gen_cycles;   // test hook: cycles of the last generation burst

    generation_controller #(
        .DATA_WIDTH(`FPGPT_DATA_WIDTH),
        .D_MODEL(`FPGPT_D_MODEL),
        .NUM_HEADS(`FPGPT_NUM_HEADS),
        .MAX_SEQ_LEN(`FPGPT_MAX_SEQ_LEN),
        .NUM_LAYERS(`FPGPT_NUM_LAYERS),
        .D_FF(`FPGPT_D_FF),
        .VOCAB_SIZE(`FPGPT_VOCAB_SIZE),
        .Q_SHIFT(`FPGPT_Q_SHIFT),
        .K_SHIFT(`FPGPT_K_SHIFT),
        .V_SHIFT(`FPGPT_V_SHIFT),
        .OUT_SHIFT(`FPGPT_OUT_SHIFT),
        .SCORE_MULT(`FPGPT_SCORE_MULT),
        .SCORE_SHIFT(`FPGPT_SCORE_SHIFT),
        .LN_SHIFT(LN_SHIFT),
        .FC1_SHIFT(`FPGPT_FC1_SHIFT),
        .FC2_SHIFT(`FPGPT_FC2_SHIFT),
        .LM_SHIFT(`FPGPT_LM_SHIFT),
        .W_ADDR_WIDTH(`FPGPT_W_ADDR_WIDTH),
        .ROM_DEPTH(`FPGPT_ROM_DEPTH),
        .ROM_MEM_FILE(ROM_MEM_FILE),
        .GEN_TOKENS(GEN_TOKENS),
        .TOP_K(TOP_K),
        .NUM_PES(NUM_PES)
    ) controller_inst (
        .clk(clk_sys),
        .rst_n(rst_n),
        .token_in(rx_data),
        .token_valid(rx_valid),
        .token_out(tx_data),
        .token_out_valid(tx_valid),
        .busy(gen_busy),
        .gen_cycles(gen_cycles)
    );

    // ── UART TX ──
    wire tx_ready;

    uart_tx #(
        .CLK_FREQ(CLK_FREQ),
        .BAUD_RATE(BAUD_RATE)
    ) tx_inst (
        .clk(clk_sys),
        .rst_n(rst_n),
        .data_in(tx_data),
        .start(tx_valid && tx_ready),
        .tx(UART_TXD),
        .ready(tx_ready)
    );

    // ── Debug LEDs ──
    // LEDR[0] = reset, LEDR[1] = PLL locked, LEDR[2] = generated token,
    // LEDR[3] = engine busy, LEDR[9:4] = last received byte[5:0].
    reg [7:0] last_rx;
    always @(posedge clk_sys) begin
        if (rx_valid) last_rx <= rx_data;
    end

    assign LEDR[0]   = ~rst_n;
    assign LEDR[1]   = pll_locked;
    assign LEDR[2]   = tx_valid;
    assign LEDR[3]   = gen_busy;
    assign LEDR[9:4] = last_rx[5:0];

    // gen_cycles is a simulation/bench hook; unused on the board.
    wire _unused = &{1'b0, gen_cycles};

endmodule
