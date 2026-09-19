`timescale 1ns/1ps

module gpt_controller #(
    parameter DATA_WIDTH = 8,
    parameter MAX_SEQ_LEN = 64
)(
    input  wire clk,
    input  wire rst_n,
    
    // Interface to host/UART
    input  wire [7:0] token_in,
    input  wire       token_valid,
    output reg  [7:0] token_out,
    output reg        token_out_valid
);

    // This is a stub for the top-level GPT controller.
    // It will eventually instantiate the transformer_block, memory interfaces,
    // and the autoregressive decoding FSM.
    
    // For now, simple loopback with a one cycle delay to test UART integration
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            token_out <= 0;
            token_out_valid <= 0;
        end else begin
            if (token_valid) begin
                // Simple rot-13 or echo for testing
                // Let's just echo back the character for the stub.
                token_out <= token_in;
                token_out_valid <= 1'b1;
            end else begin
                token_out_valid <= 1'b0;
            end
        end
    end

endmodule
