`timescale 1ns/1ps

module uart_rx #(
    parameter CLK_FREQ = 50000000,
    parameter BAUD_RATE = 115200
)(
    input  wire       clk,
    input  wire       rst_n,
    input  wire       rx,
    output reg  [7:0] data_out,
    output reg        valid
);

    localparam CLOCKS_PER_BIT = CLK_FREQ / BAUD_RATE;
    
    localparam S_IDLE  = 3'd0;
    localparam S_START = 3'd1;
    localparam S_DATA  = 3'd2;
    localparam S_STOP  = 3'd3;
    localparam S_CLEAN = 3'd4;
    
    reg [2:0] state;
    reg [$clog2(CLOCKS_PER_BIT)-1:0] clk_count;
    reg [2:0] bit_index;
    reg [7:0] shift_reg;
    
    // Double-flop RX for metastability
    reg rx_q1, rx_q2;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rx_q1 <= 1'b1;
            rx_q2 <= 1'b1;
        end else begin
            rx_q1 <= rx;
            rx_q2 <= rx_q1;
        end
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state     <= S_IDLE;
            clk_count <= 0;
            bit_index <= 0;
            shift_reg <= 0;
            valid     <= 1'b0;
            data_out  <= 0;
        end else begin
            valid <= 1'b0;
            
            case (state)
                S_IDLE: begin
                    clk_count <= 0;
                    bit_index <= 0;
                    if (rx_q2 == 1'b0) begin
                        state <= S_START;
                    end
                end
                
                S_START: begin
                    if (clk_count == (CLOCKS_PER_BIT - 1) / 2) begin
                        if (rx_q2 == 1'b0) begin
                            clk_count <= 0;
                            state     <= S_DATA;
                        end else begin
                            state     <= S_IDLE;
                        end
                    end else begin
                        clk_count <= clk_count + 1'b1;
                    end
                end
                
                S_DATA: begin
                    if (clk_count < CLOCKS_PER_BIT - 1) begin
                        clk_count <= clk_count + 1'b1;
                    end else begin
                        clk_count <= 0;
                        shift_reg[bit_index] <= rx_q2;
                        if (bit_index < 7) begin
                            bit_index <= bit_index + 1'b1;
                        end else begin
                            state <= S_STOP;
                        end
                    end
                end
                
                S_STOP: begin
                    if (clk_count < CLOCKS_PER_BIT - 1) begin
                        clk_count <= clk_count + 1'b1;
                    end else begin
                        // Wait for stop bit
                        if (rx_q2 == 1'b1) begin
                            data_out <= shift_reg;
                            valid    <= 1'b1;
                            state    <= S_CLEAN;
                        end else begin
                            // Framing error
                            state    <= S_IDLE;
                        end
                    end
                end
                
                S_CLEAN: begin
                    state <= S_IDLE;
                end
                
                default: state <= S_IDLE;
            endcase
        end
    end
endmodule
