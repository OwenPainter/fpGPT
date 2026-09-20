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
`define FPGPT_VOCAB_SIZE      98

// ── Weight ROM (uniform-width image consumed by transformer_engine) ──
`define FPGPT_W_ADDR_WIDTH    18
`define FPGPT_ROM_DEPTH       216704
`define FPGPT_ROM_MEM_FILE    "weights_unified.hex"

// ── Decode configuration ──
`define FPGPT_GEN_TOKENS      16
`define FPGPT_NUM_PES         4

// ── Fixed-point shifts (layer 0; engine uses one global value) ──
`define FPGPT_Q_SHIFT         6
`define FPGPT_K_SHIFT         6
`define FPGPT_V_SHIFT         7
`define FPGPT_OUT_SHIFT       6
`define FPGPT_SCORE_MULT      262144
`define FPGPT_SCORE_SHIFT     26
`define FPGPT_LN_SHIFT        7
`define FPGPT_FC1_SHIFT       6
`define FPGPT_FC2_SHIFT       6
`define FPGPT_LM_SHIFT        4

// ── Board / host interface ──
`define FPGPT_SYS_CLK_HZ      62500000
`define FPGPT_BAUD_RATE       115200

// B0: Q=6 K=6 V=7 OUT=6 FC1=6 FC2=6
// B1: Q=7 K=6 V=7 OUT=7 FC1=6 FC2=6
// B2: Q=6 K=6 V=7 OUT=7 FC1=6 FC2=6
// B3: Q=6 K=6 V=7 OUT=6 FC1=6 FC2=4

`endif // FPGPT_BOARD_PARAMS_VH
