// ═══════════════════════════════════════════════════════════════
// fpGPT — Synchronous ROM (M10K Block RAM Inference)
// Target: Intel Cyclone V M10K embedded memory blocks
//
// This module is written to follow Quartus synthesis guidelines
// for automatic M10K block RAM inference:
//   - Single-port synchronous read
//   - Registered output
//   - Initialized via $readmemh
//
// Each M10K block provides 10,240 bits = 1,280 bytes.
// The Cyclone V 5CSEMA5F31C6 has 397 M10K blocks = ~556 KB total.
// ═══════════════════════════════════════════════════════════════

module rom_sync #(
    parameter DATA_WIDTH = 8,
    parameter ADDR_WIDTH = 16,
    parameter DEPTH      = 65536,
    parameter MEM_FILE   = "weights.hex"   // $readmemh initialization file
) (
    input  wire                    clk,
    input  wire [ADDR_WIDTH-1:0]  addr,
    output reg  [DATA_WIDTH-1:0]  data
);

    // Inferred M10K block RAM
    // Quartus recognizes this pattern and maps it to M10K blocks
    (* ramstyle = "M10K" *) reg [DATA_WIDTH-1:0] mem [0:DEPTH-1];

    // Load weights from hex file at synthesis/simulation time
    initial begin
        $readmemh(MEM_FILE, mem);
    end

    // Synchronous read (1-cycle latency)
    always @(posedge clk) begin
        data <= mem[addr];
    end

endmodule
