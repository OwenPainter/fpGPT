`timescale 1ns/1ps

// ═══════════════════════════════════════════════════════════════
// fpGPT — DE1-SoC Top Level Wrapper
// Connects the GPT controller, UART, and physical DE1-SoC pins.
// ═══════════════════════════════════════════════════════════════

module fpga_top (
    // Clock & Reset
    input  wire        CLOCK_50,
    input  wire [3:0]  KEY,      // KEY[0] is active-low reset
    
    // UART interface
    // Standard DE1-SoC UART pins (assumed PIN_AH9 for TX, PIN_AG11 for RX)
    // Wait, the QSF will map these
    input  wire        UART_RXD,
    output wire        UART_TXD,
    
    // Debug
    output wire [9:0]  LEDR
);

    wire clk = CLOCK_50;
    wire rst_n = KEY[0];
    
    // ── UART RX ──
    wire [7:0] rx_data;
    wire       rx_valid;
    
    uart_rx #(
        .CLK_FREQ(50000000),
        .BAUD_RATE(115200)
    ) rx_inst (
        .clk(clk),
        .rst_n(rst_n),
        .rx(UART_RXD),
        .data_out(rx_data),
        .valid(rx_valid)
    );
    
    // ── GPT Controller ──
    wire [7:0] tx_data;
    wire       tx_valid;
    
    gpt_controller #(
        .DATA_WIDTH(8),
        .MAX_SEQ_LEN(64)
    ) controller_inst (
        .clk(clk),
        .rst_n(rst_n),
        .token_in(rx_data),
        .token_valid(rx_valid),
        .token_out(tx_data),
        .token_out_valid(tx_valid)
    );
    
    // ── UART TX ──
    wire tx_ready;
    
    // Simple FIFO / buffer might be needed if the controller outputs faster
    // than UART can send. For the stub (echo), we assume it's slow enough 
    // or we just drop tokens if not ready. A proper FIFO is needed in final.
    uart_tx #(
        .CLK_FREQ(50000000),
        .BAUD_RATE(115200)
    ) tx_inst (
        .clk(clk),
        .rst_n(rst_n),
        .data_in(tx_data),
        .start(tx_valid && tx_ready), // only start if ready
        .tx(UART_TXD),
        .ready(tx_ready)
    );
    
    // ── Debug LEDs ──
    // LEDR[0] = reset state
    // LEDR[1] = UART RX active (blinks on valid)
    // LEDR[2] = UART TX active (blinks on valid)
    // LEDR[9:3] = last received token
    assign LEDR[0] = ~rst_n;
    assign LEDR[1] = rx_valid;
    assign LEDR[2] = tx_valid;
    
    reg [6:0] last_token;
    always @(posedge clk) begin
        if (rx_valid) last_token <= rx_data[6:0];
    end
    assign LEDR[9:3] = last_token;

endmodule
