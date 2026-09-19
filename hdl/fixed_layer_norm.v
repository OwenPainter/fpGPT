// Sequential one-vector LayerNorm. Gamma Q14/int32, beta OUT_FRAC/int32.
// Statistics retain eight extra fractional bits. See docs/fixed_point.md.
module fixed_layer_norm #(
    parameter DATA_WIDTH=8, DIM=64, OUT_FRAC=4,
    parameter signed [63:0] EPSILON_INT=671,
    parameter ADDR_WIDTH=(DIM <= 1 ? 1 : $clog2(DIM))
) (
    input wire clk, rst_n, load, start,
    input wire [ADDR_WIDTH-1:0] load_addr,
    input wire signed [DATA_WIDTH-1:0] x_data,
    input wire signed [31:0] gamma_data, beta_data,
    output reg busy, done, y_valid,
    output reg [ADDR_WIDTH-1:0] y_addr,
    output reg signed [DATA_WIDTH-1:0] y_data
);
    localparam IDLE=0, SUM=1, MEAN=2, VARIANCE=3, ROOT=4, NORMALIZE=5, FINISH=6;
    reg [2:0] state;
    reg signed [DATA_WIDTH-1:0] x[0:DIM-1];
    reg signed [31:0] gamma[0:DIM-1], beta[0:DIM-1];
    integer index;
    reg signed [63:0] total, mean, variance, stddev;
    wire signed [63:0] x_wide = x[index];
    wire signed [63:0] centered = (x_wide <<< 8)-mean;
    wire signed [63:0] normalized = (centered <<< 14)/stddev;
    wire signed [63:0] affine = ((normalized * gamma[index]) >>> (28-OUT_FRAC)) + beta[index];
    localparam signed [63:0] MAXIMUM=(64'sd1 << (DATA_WIDTH-1))-1;
    localparam signed [63:0] MINIMUM=-(64'sd1 << (DATA_WIDTH-1));

    // Fixed 32-step restoring square root; integer floor, synthesizable.
    function [63:0] isqrt;
        input [63:0] value;
        reg [63:0] remainder, root, bit_value;
        integer i;
        begin
            remainder=value; root=0; bit_value=64'h4000000000000000;
            for(i=0; i<32; i=i+1) begin
                if(remainder >= root+bit_value) begin
                    remainder=remainder-root-bit_value;
                    root=(root >> 1)+bit_value;
                end else root=root >> 1;
                bit_value=bit_value >> 2;
            end
            isqrt=root;
        end
    endfunction

    wire signed [63:0] stddev_next = isqrt(variance/DIM+EPSILON_INT);

    always @(posedge clk) if(rst_n && !busy && load && load_addr < DIM) begin
        x[load_addr] <= x_data; gamma[load_addr] <= gamma_data; beta[load_addr] <= beta_data;
    end
    always @(posedge clk or negedge rst_n) begin
        if(!rst_n) begin
            state<=IDLE; busy<=0; done<=0; y_valid<=0; y_addr<=0; y_data<=0;
            index<=0; total<=0; mean<=0; variance<=0; stddev<=1;
        end else begin
            done<=0; y_valid<=0;
            case(state)
                IDLE: if(start) begin busy<=1; total<=0; index<=0; state<=SUM; end
                SUM: begin
                    total<=total+(x_wide <<< 8);
                    if(index==DIM-1) state<=MEAN; else index<=index+1;
                end
                MEAN: begin mean<=total/DIM; index<=0; variance<=0; state<=VARIANCE; end
                VARIANCE: begin
                    variance<=variance+centered*centered;
                    if(index==DIM-1) state<=ROOT; else index<=index+1;
                end
                ROOT: begin stddev<=stddev_next; index<=0; state<=NORMALIZE; end
                NORMALIZE: begin
                    y_data<=affine > MAXIMUM ? MAXIMUM[DATA_WIDTH-1:0] :
                            affine < MINIMUM ? MINIMUM[DATA_WIDTH-1:0] : affine[DATA_WIDTH-1:0];
                    y_addr<=index; y_valid<=1;
                    if(index==DIM-1) state<=FINISH; else index<=index+1;
                end
                FINISH: begin busy<=0; done<=1; state<=IDLE; end
                default: begin state<=IDLE; busy<=0; end
            endcase
        end
    end
endmodule
