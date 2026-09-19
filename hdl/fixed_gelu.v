// Compiler-generated signed GELU lookup table, indexed by input bit pattern.
module fixed_gelu #(
    parameter DATA_WIDTH=8, MEM_FILE="gelu.hex"
) (
    input wire clk,
    input wire [DATA_WIDTH-1:0] x,
    output reg signed [DATA_WIDTH-1:0] y
);
    reg signed [DATA_WIDTH-1:0] table_data[0:(1 << DATA_WIDTH)-1];
    initial $readmemh(MEM_FILE, table_data);
    always @(posedge clk) y <= table_data[x];
endmodule
