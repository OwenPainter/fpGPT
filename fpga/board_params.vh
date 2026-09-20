// ===================================================
// fpGPT Compiler -- Auto-generated Board Parameters
// DO NOT EDIT -- regenerate with: python compile.py
// Consumed by fpga/fpga_top.v; fpga/build.tcl copies it in.
// ===================================================

`ifndef FPGPT_BOARD_PARAMS_VH
`define FPGPT_BOARD_PARAMS_VH

// ── Model dimensions ──
`define FPGPT_DATA_WIDTH      8
`define FPGPT_D_MODEL         64
`define FPGPT_NUM_HEADS       4
`define FPGPT_MAX_SEQ_LEN     64
`define FPGPT_NUM_LAYERS      4
`define FPGPT_D_FF            256
`define FPGPT_VOCAB_SIZE      64

// ── Weight ROM (uniform-width image consumed by transformer_engine) ──
`define FPGPT_W_ADDR_WIDTH    18
`define FPGPT_ROM_DEPTH       212352
`define FPGPT_ROM_MEM_FILE    "weights_unified.hex"

// ── Decode configuration ──
`define FPGPT_GEN_TOKENS      16

// Output features computed concurrently per PE-array pass. Must be a power of
// two and divide D_FF, D_MODEL and VOCAB_SIZE (see hdl/weight_cache.v). 1 keeps
// the original single-byte ROM path.
`define FPGPT_NUM_PES         4

// ── Fixed-point shifts (layer 0; engine uses one global value) ──
`define FPGPT_Q_SHIFT         6
`define FPGPT_K_SHIFT         6
`define FPGPT_V_SHIFT         7
`define FPGPT_OUT_SHIFT       7
`define FPGPT_SCORE_MULT      262144
`define FPGPT_SCORE_SHIFT     28
`define FPGPT_LN_SHIFT        7
`define FPGPT_FC1_SHIFT       6
`define FPGPT_FC2_SHIFT       8
`define FPGPT_LM_SHIFT        6

// ── Board / host interface ──
`define FPGPT_SYS_CLK_HZ      62500000
`define FPGPT_BAUD_RATE       115200

// B0: Q=6 K=6 V=7 OUT=7 FC1=6 FC2=8
// B1: Q=7 K=7 V=7 OUT=8 FC1=6 FC2=7
// B2: Q=7 K=7 V=6 OUT=7 FC1=6 FC2=7
// B3: Q=7 K=6 V=6 OUT=7 FC1=6 FC2=6

`endif // FPGPT_BOARD_PARAMS_VH
