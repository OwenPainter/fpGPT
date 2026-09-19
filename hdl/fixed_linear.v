// Sequential affine transform with signed 64-bit accumulator-domain biases.
// Load x, row-major weights and biases while idle; pulse start; capture y_valid.
module fixed_linear #(
    parameter DATA_WIDTH=8, IN_FEATURES=64, OUT_FEATURES=64, SHIFT=7,
    parameter X_ADDR_WIDTH=(IN_FEATURES<=1 ? 1 : $clog2(IN_FEATURES)),
    parameter Y_ADDR_WIDTH=(OUT_FEATURES<=1 ? 1 : $clog2(OUT_FEATURES)),
    parameter W_ADDR_WIDTH=(IN_FEATURES*OUT_FEATURES<=1 ? 1 : $clog2(IN_FEATURES*OUT_FEATURES))
) (
    input wire clk, rst_n, start, x_load, w_load, b_load,
    input wire [X_ADDR_WIDTH-1:0] x_addr,
    input wire [W_ADDR_WIDTH-1:0] w_addr,
    input wire [Y_ADDR_WIDTH-1:0] b_addr,
    input wire signed [DATA_WIDTH-1:0] x_data,w_data,
    input wire signed [63:0] b_data,
    output reg busy,done,y_valid,
    output reg [Y_ADDR_WIDTH-1:0] y_addr,
    output reg signed [DATA_WIDTH-1:0] y_data
);
    localparam IDLE=0, READ=1, MAC=2, SAVE=3, FINISH=4;
    reg [2:0] state;
    reg signed [DATA_WIDTH-1:0] inputs[0:IN_FEATURES-1];
    reg signed [DATA_WIDTH-1:0] weights[0:IN_FEATURES*OUT_FEATURES-1];
    reg signed [63:0] biases[0:OUT_FEATURES-1];
    reg signed [DATA_WIDTH-1:0] a,b;
    wire signed [2*DATA_WIDTH-1:0] product=a*b;
    reg signed [63:0] acc;
    integer row,col;
    wire signed [63:0] biased=acc+biases[row];
    wire signed [63:0] scaled=SHIFT >= 0 ? biased >>> SHIFT : biased <<< -SHIFT;
    localparam signed [63:0] MAXIMUM=(64'sd1 << (DATA_WIDTH-1))-1;
    localparam signed [63:0] MINIMUM=-(64'sd1 << (DATA_WIDTH-1));
    always @(posedge clk) begin
        if(rst_n && !busy) begin
            if(x_load && x_addr<IN_FEATURES) inputs[x_addr]<=x_data;
            if(w_load && w_addr<IN_FEATURES*OUT_FEATURES) weights[w_addr]<=w_data;
            if(b_load && b_addr<OUT_FEATURES) biases[b_addr]<=b_data;
        end
        if(state==READ) begin a<=inputs[col]; b<=weights[row*IN_FEATURES+col]; end
    end
    always @(posedge clk or negedge rst_n) begin
        if(!rst_n) begin
            state<=IDLE; busy<=0; done<=0; y_valid<=0; y_addr<=0; y_data<=0; acc<=0; row<=0; col<=0;
        end else begin
            done<=0; y_valid<=0;
            case(state)
                IDLE: if(start) begin busy<=1; row<=0; col<=0; acc<=0; state<=READ; end
                READ: state<=MAC;
                MAC: begin
                    acc<=acc+product;
                    if(col==IN_FEATURES-1) state<=SAVE;
                    else begin col<=col+1; state<=READ; end
                end
                SAVE: begin
                    y_data<=scaled>MAXIMUM ? MAXIMUM[DATA_WIDTH-1:0] :
                            scaled<MINIMUM ? MINIMUM[DATA_WIDTH-1:0] : scaled[DATA_WIDTH-1:0];
                    y_addr<=row; y_valid<=1; col<=0; acc<=0;
                    if(row==OUT_FEATURES-1) state<=FINISH;
                    else begin row<=row+1; state<=READ; end
                end
                FINISH: begin done<=1; busy<=0; state<=IDLE; end
                default: begin state<=IDLE; busy<=0; end
            endcase
        end
    end
endmodule
