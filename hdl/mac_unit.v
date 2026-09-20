// ═══════════════════════════════════════════════════════════════
// fpGPT — Parameterized Multiply-Accumulate (MAC) Unit
// Target: Intel Cyclone V DSP blocks (18x18-bit multipliers)
//
// Performs: accumulator += weight * activation
// Pipeline: 2-stage (multiply → accumulate) for Fmax optimization
//
// The Cyclone V has 87 DSP blocks, each containing one 18x18
// signed multiplier. This module is designed to map cleanly to
// a single DSP block when DATA_WIDTH <= 18.
// ═══════════════════════════════════════════════════════════════

module mac_unit #(
    parameter DATA_WIDTH  = 8,    // Input operand width (INT8)
    parameter ACC_WIDTH   = 32    // Accumulator width (prevents overflow)
) (
    input  wire                    clk,
    input  wire                    rst_n,       // Active-low reset
    input  wire                    clear,       // Clear accumulator to zero
    input  wire                    enable,      // Enable MAC operation
    input  wire signed [DATA_WIDTH-1:0]  weight,     // Weight input (from ROM)
    input  wire signed [DATA_WIDTH-1:0]  activation, // Activation input
    output reg  signed [ACC_WIDTH-1:0]   accumulator // Running sum
);

    // ── Pipeline stage 1: register the product ──
    // The product register maps into the DSP block's output register, which
    // keeps the multiplier off the accumulator's timing path.
    reg signed [2*DATA_WIDTH-1:0] product_reg;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            product_reg <= {2*DATA_WIDTH{1'b0}};
        end else if (clear) begin
            product_reg <= {2*DATA_WIDTH{1'b0}};
        end else if (enable) begin
            product_reg <= weight * activation;
        end
    end

    // ── Pipeline stage 2: accumulate the registered product ──
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            accumulator <= {ACC_WIDTH{1'b0}};
        end else if (clear) begin
            accumulator <= {ACC_WIDTH{1'b0}};
        end else if (enable) begin
            // Sign-extend the registered product to accumulator width
            accumulator <= accumulator + {{(ACC_WIDTH-2*DATA_WIDTH){product_reg[2*DATA_WIDTH-1]}}, product_reg};
        end
    end

endmodule
