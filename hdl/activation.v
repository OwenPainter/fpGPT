// ═══════════════════════════════════════════════════════════════
// fpGPT — Hardware Activation Functions
// Target: Intel Cyclone V (pure combinational / LUT-based)
//
// Implements:
//   - ReLU:       max(0, x)
//   - GELU approx: x * sigmoid(1.702 * x) via piecewise LUT
//   - Clamp/Saturate: Clamp accumulator to DATA_WIDTH output range
// ═══════════════════════════════════════════════════════════════

module activation #(
    parameter DATA_WIDTH = 8,          // Output width
    parameter ACC_WIDTH  = 32,         // Input accumulator width
    parameter SHIFT_BITS = 7,          // Right-shift for de-quantization
    parameter MODE       = "relu"      // "relu", "gelu", "none"
) (
    input  wire signed [ACC_WIDTH-1:0]  acc_in,    // Raw accumulator value
    output wire signed [DATA_WIDTH-1:0] data_out   // Quantized activated output
);

    // ── Step 1: Scale down the accumulator ──
    // The accumulator holds: sum(w_i * x_i) where w and x are INT8.
    // We right-shift to bring it back to INT8 range.
    wire signed [ACC_WIDTH-1:0] scaled;
    assign scaled = acc_in >>> SHIFT_BITS;

    // ── Step 2: Apply activation function ──
    wire signed [ACC_WIDTH-1:0] activated;

    generate
        if (MODE == "relu") begin : gen_relu
            // ReLU: output = (x < 0) ? 0 : x
            assign activated = (scaled[ACC_WIDTH-1]) ? {ACC_WIDTH{1'b0}} : scaled;
        end
        else if (MODE == "gelu") begin : gen_gelu
            // GELU approximation via piecewise linear:
            //   x <= -3:  output = 0
            //   -3 < x < 0:  output = x/4  (rough lower sigmoid region)
            //   0 <= x < 3:  output = 3x/4 (rough upper sigmoid region)
            //   x >= 3:   output = x
            // This is a coarse but hardware-efficient approximation.
            wire signed [ACC_WIDTH-1:0] abs_x;
            assign abs_x = scaled[ACC_WIDTH-1] ? -scaled : scaled;

            reg signed [ACC_WIDTH-1:0] gelu_result;
            always @(*) begin
                if (scaled <= -3 * (1 << (DATA_WIDTH-3))) begin
                    gelu_result = {ACC_WIDTH{1'b0}};  // ≈ 0 for very negative
                end else if (scaled < 0) begin
                    gelu_result = scaled >>> 2;        // ≈ x/4
                end else if (scaled < 3 * (1 << (DATA_WIDTH-3))) begin
                    gelu_result = (scaled * 3) >>> 2;  // ≈ 3x/4
                end else begin
                    gelu_result = scaled;              // ≈ x for very positive
                end
            end
            assign activated = gelu_result;
        end
        else begin : gen_none
            // No activation — pass through
            assign activated = scaled;
        end
    endgenerate

    // ── Step 3: Saturating clamp to DATA_WIDTH range ──
    // Clamp to [-128, 127] for INT8
    localparam signed [DATA_WIDTH-1:0] MAX_VAL =  (1 << (DATA_WIDTH-1)) - 1;  // 127
    localparam signed [DATA_WIDTH-1:0] MIN_VAL = -(1 << (DATA_WIDTH-1));       // -128

    wire overflow_pos = (~activated[ACC_WIDTH-1]) && (activated > {{(ACC_WIDTH-DATA_WIDTH){1'b0}}, MAX_VAL});
    wire overflow_neg = (activated[ACC_WIDTH-1])  && (activated < {{(ACC_WIDTH-DATA_WIDTH){1'b1}}, MIN_VAL});

    assign data_out = overflow_pos ? MAX_VAL :
                      overflow_neg ? MIN_VAL :
                      activated[DATA_WIDTH-1:0];

endmodule
