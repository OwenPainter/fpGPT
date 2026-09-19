// Align at the finest scale, add with headroom, then shift and saturate once.
module fixed_residual #(
    parameter DATA_WIDTH=8, A_FRAC=5, B_FRAC=5, OUT_FRAC=5
) (
    input wire signed [DATA_WIDTH-1:0] a, b,
    output wire signed [DATA_WIDTH-1:0] y
);
    localparam AB_FRAC = A_FRAC > B_FRAC ? A_FRAC : B_FRAC;
    localparam COMMON = AB_FRAC > OUT_FRAC ? AB_FRAC : OUT_FRAC;
    wire signed [63:0] wide_a = a;
    wire signed [63:0] wide_b = b;
    wire signed [63:0] total = (wide_a <<< (COMMON-A_FRAC)) + (wide_b <<< (COMMON-B_FRAC));
    wire signed [63:0] scaled = total >>> (COMMON-OUT_FRAC);
    localparam signed [63:0] MAXIMUM = (64'sd1 << (DATA_WIDTH-1))-1;
    localparam signed [63:0] MINIMUM = -(64'sd1 << (DATA_WIDTH-1));
    assign y = scaled > MAXIMUM ? MAXIMUM[DATA_WIDTH-1:0] :
               scaled < MINIMUM ? MINIMUM[DATA_WIDTH-1:0] : scaled[DATA_WIDTH-1:0];
endmodule
