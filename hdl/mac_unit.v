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

    // ── Pipeline stage 1: Multiply ──
    // Use full-width product to avoid overflow in the multiplier itself
    wire signed [2*DATA_WIDTH-1:0] product;
    assign product = weight * activation;

    // ── Pipeline stage 2: Accumulate ──
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            accumulator <= {ACC_WIDTH{1'b0}};
        end else if (clear) begin
            accumulator <= {ACC_WIDTH{1'b0}};
        end else if (enable) begin
            // Sign-extend the product to accumulator width before adding
            accumulator <= accumulator + {{(ACC_WIDTH-2*DATA_WIDTH){product[2*DATA_WIDTH-1]}}, product};
        end
    end

endmodule
