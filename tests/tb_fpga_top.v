`timescale 1ns/1ps

module tb_fpga_top;

    reg clk;
    reg rst_n;
    reg uart_rx;
    wire uart_tx;
    wire [9:0] ledr;

    // Instantiate top-level
    fpga_top dut (
        .CLOCK_50(clk),
        .KEY({3'b111, rst_n}),
        .UART_RXD(uart_rx),
        .UART_TXD(uart_tx),
        .LEDR(ledr)
    );

    // Clock generation (50MHz)
    always #10 clk = ~clk;

    localparam CLOCKS_PER_BIT = 50000000 / 115200;
    localparam BIT_PERIOD = CLOCKS_PER_BIT * 20; // 20ns clock period
    
    // Task to send a UART byte to the FPGA
    task send_uart_byte;
        input [7:0] data;
        integer i;
        begin
            // Start bit
            uart_rx = 1'b0;
            #(BIT_PERIOD);
            
            // Data bits (LSB first)
            for (i = 0; i < 8; i = i + 1) begin
                uart_rx = data[i];
                #(BIT_PERIOD);
            end
            
            // Stop bit
            uart_rx = 1'b1;
            #(BIT_PERIOD);
        end
    endtask

    // Variable to capture TX bits
    reg [7:0] rx_byte;

    // Task to receive a UART byte from the FPGA
    task receive_uart_byte;
        output [7:0] data;
        integer i;
        begin
            // Wait for start bit
            wait (uart_tx == 1'b0);
            #(BIT_PERIOD / 2); // Sample at middle of start bit
            #(BIT_PERIOD);     // Move to first data bit
            
            for (i = 0; i < 8; i = i + 1) begin
                data[i] = uart_tx;
                #(BIT_PERIOD);
            end
            
            // Wait for stop bit (should be high)
            if (uart_tx !== 1'b1) begin
                $display("Warning: UART TX framing error (stop bit missing)");
            end
        end
    endtask

    initial begin
        // Initialize
        clk = 0;
        rst_n = 0;
        uart_rx = 1'b1; // Idle high
        
        // Reset
        #100 rst_n = 1;
        #100;
        
        // Send a character 'A' (8'h41)
        $display("Sending 'A' (0x41) to fpga_top over UART RX...");
        send_uart_byte(8'h41);
        
        // Wait and receive the echoed response from gpt_controller stub
        receive_uart_byte(rx_byte);
        $display("Received response from UART TX: 0x%0h", rx_byte);
        
        if (rx_byte == 8'h41) begin
            $display("TEST PASSED: Echo loopback successful!");
        end else begin
            $display("TEST FAILED: Expected 0x41, got 0x%0h", rx_byte);
        end
        
        #1000 $finish;
    end

endmodule
