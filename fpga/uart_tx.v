`timescale 1ns/1ps

module uart_tx #(
    parameter CLK_FREQ = 50000000,
    parameter BAUD_RATE = 115200
)(
    input  wire       clk,
    input  wire       rst_n,
    input  wire [7:0] data_in,
    input  wire       start,
    output reg        tx,
    output reg        ready
);

    localparam CLOCKS_PER_BIT = CLK_FREQ / BAUD_RATE;
    
    localparam S_IDLE  = 2'd0;
    localparam S_START = 2'd1;
    localparam S_DATA  = 2'd2;
    localparam S_STOP  = 2'd3;
    
    reg [1:0] state;
    reg [$clog2(CLOCKS_PER_BIT)-1:0] clk_count;
    reg [2:0] bit_index;
    reg [7:0] shift_reg;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state     <= S_IDLE;
            tx        <= 1'b1;
            ready     <= 1'b1;
            clk_count <= 0;
            bit_index <= 0;
            shift_reg <= 0;
        end else begin
            case (state)
                S_IDLE: begin
                    tx <= 1'b1;
                    if (start) begin
                        ready     <= 1'b0;
                        shift_reg <= data_in;
                        state     <= S_START;
                        clk_count <= 0;
                    end else begin
                        ready <= 1'b1;
                    end
                end
                
                S_START: begin
                    tx <= 1'b0;
                    if (clk_count < CLOCKS_PER_BIT - 1) begin
                        clk_count <= clk_count + 1'b1;
                    end else begin
                        clk_count <= 0;
                        state     <= S_DATA;
                        bit_index <= 0;
                    end
                end
                
                S_DATA: begin
                    tx <= shift_reg[bit_index];
                    if (clk_count < CLOCKS_PER_BIT - 1) begin
                        clk_count <= clk_count + 1'b1;
                    end else begin
                        clk_count <= 0;
                        if (bit_index < 7) begin
                            bit_index <= bit_index + 1'b1;
                        end else begin
                            state <= S_STOP;
                        end
                    end
                end
                
                S_STOP: begin
                    tx <= 1'b1;
                    if (clk_count < CLOCKS_PER_BIT - 1) begin
                        clk_count <= clk_count + 1'b1;
                    end else begin
                        clk_count <= 0;
                        state     <= S_IDLE;
                        ready     <= 1'b1;
                    end
                end
                
                default: state <= S_IDLE;
            endcase
        end
    end
endmodule
