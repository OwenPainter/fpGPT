`timescale 1ns/1ps

module tb_transformer_block;

    // Parameters matching the Python test model
    localparam DATA_WIDTH = 8;
    localparam D_MODEL = 8;
    localparam MAX_SEQ_LEN = 4;
    
    reg clk;
    reg rst_n;
    reg enable;
    
    // Inputs (streaming sequentially for now)
    reg signed [DATA_WIDTH-1:0] x_in [0:D_MODEL-1];
    
    // Outputs
    wire signed [DATA_WIDTH-1:0] x_out [0:D_MODEL-1];
    wire done;
    
    // Instantiate the DUT (Device Under Test) stub
    transformer_block #(
        .DATA_WIDTH(DATA_WIDTH),
        .D_MODEL(D_MODEL),
        .MAX_SEQ_LEN(MAX_SEQ_LEN)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .enable(enable),
        // Simplification for the stub: passing whole vectors or memory interfaces
        // For a true implementation, memory interfaces or streams would be used.
        // We'll leave the port map generic for the stub.
        .done(done)
    );
    
    // Clock generation
    always #5 clk = ~clk;
    
    initial begin
        // Initialize
        clk = 0;
        rst_n = 0;
        enable = 0;
        
        // Reset
        #20 rst_n = 1;
        
        // Load stimulus (if we had the input arrays explicitly defined)
        // $readmemh("build/test_rtl/tb_input.hex", test_memory);
        
        // Start processing
        #10 enable = 1;
        #10 enable = 0;
        
        // Wait for done
        wait(done);
        
        // Output validation would happen here or dumped to file
        // $writememh("build/test_rtl/tb_output.hex", output_memory);
        
        #50 $finish;
    end
    
endmodule
