"""
fpGPT Compiler — CLI Entry Point

End-to-end pipeline: Load trained PyTorch model → Quantize to INT8 →
Export .mif weight files → Generate synthesizable Verilog RTL.

Usage:
    python compile.py --model checkpoints/micro_gpt.pt --out build/
    python compile.py --model checkpoints/micro_gpt.pt --out build/ --precision 16
"""

import argparse
import os
import sys
import time

import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model.micro_gpt import MicroGPT
from compiler.quantizer import quantize_model
from compiler.mif_writer import export_all_weights


def compile_model(args):
    print("+======================================================+")
    print("|         fpGPT Compiler -- ML Weights to Verilog      |")
    print("+======================================================+")
    print()

    t0 = time.time()

    # ── Step 1: Load trained model ──
    print("[1/4] Loading trained model...")
    checkpoint = torch.load(args.model, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    model = MicroGPT(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"       Loaded model with {model.param_count:,} parameters")
    print(f"       Architecture: d={config['d_model']}, "
          f"heads={config['num_heads']}, layers={config['num_layers']}, "
          f"d_ff={config['d_ff']}")

    # ── Step 2: Quantize ──
    print(f"\n[2/4] Quantizing to INT{args.precision}...")
    ir = quantize_model(model, config, bit_width=args.precision)
    print(f"       Total parameters: {ir.total_params:,}")
    print(f"       Weight memory:    {ir.total_weight_bytes:,} bytes "
          f"({ir.total_weight_bytes / 1024:.1f} KB)")
    budget_pct = ir.total_weight_bytes / (556 * 1024) * 100
    print(f"       M10K budget:      {budget_pct:.1f}% of 556 KB on-chip BRAM")

    if budget_pct > 100:
        print(f"\n  ⚠ WARNING: Model exceeds on-chip M10K capacity!")
        print(f"    Consider reducing d_model, num_layers, or using INT4 quantization.")
        print(f"    Alternatively, use external SDRAM for weight storage.")

    # ── Step 3: Export weight files ──
    weights_dir = os.path.join(args.out, "weights")
    print(f"\n[3/4] Exporting weight files to {weights_dir}/")
    exported = export_all_weights(ir, weights_dir, fmt=args.fmt)
    print(f"       Generated {len(exported)} weight files")
    total_bytes = sum(e[2] for e in exported)
    print(f"       Total weight data: {total_bytes:,} bytes")

    # ── Step 4: Generate Verilog RTL ──
    rtl_dir = os.path.join(args.out, "rtl")
    os.makedirs(rtl_dir, exist_ok=True)
    print(f"\n[4/4] Generating Verilog RTL to {rtl_dir}/")

    # Generate parameters header
    params_path = os.path.join(rtl_dir, "gpt_params.vh")
    _write_params_header(ir, params_path)
    print(f"       [ok] {params_path}")

    # Generate ROM instantiation wrapper
    rom_path = os.path.join(rtl_dir, "weight_rom.v")
    _write_weight_rom(ir, rom_path)
    print(f"       [ok] {rom_path}")

    elapsed = time.time() - t0
    print(f"\n{'=' * 56}")
    print(f"  Compilation complete in {elapsed:.2f}s")
    print(f"  Output directory: {os.path.abspath(args.out)}")
    print(f"{'=' * 56}")

    # Print model IR summary
    print(f"\n{ir.summary()}")


def _write_params_header(ir, filepath: str):
    """Generate a Verilog parameters header (.vh) with model constants."""
    with open(filepath, 'w') as f:
        f.write("// ===================================================\n")
        f.write("// fpGPT Compiler -- Auto-generated Model Parameters\n")
        f.write("// DO NOT EDIT -- regenerate with: python compile.py\n")
        f.write("// ===================================================\n\n")

        f.write(f"// Architecture\n")
        f.write(f"parameter VOCAB_SIZE    = {ir.vocab_size};\n")
        f.write(f"parameter MAX_SEQ_LEN   = {ir.max_seq_len};\n")
        f.write(f"parameter D_MODEL       = {ir.d_model};\n")
        f.write(f"parameter NUM_HEADS     = {ir.num_heads};\n")
        f.write(f"parameter HEAD_DIM      = {ir.d_model // ir.num_heads};\n")
        f.write(f"parameter NUM_LAYERS    = {ir.num_layers};\n")
        f.write(f"parameter D_FF          = {ir.d_ff};\n")
        f.write(f"parameter BIT_WIDTH     = {ir.bit_width};\n\n")

        f.write(f"// Memory layout\n")
        f.write(f"parameter TOTAL_WEIGHT_BYTES = {ir.total_weight_bytes};\n\n")

        # Per-layer base addresses
        f.write(f"// Embedding base addresses\n")
        if ir.token_embedding:
            f.write(f"parameter TOK_EMB_ADDR  = {ir.token_embedding.weight_base_addr};\n")
            f.write(f"parameter TOK_EMB_SIZE  = {ir.token_embedding.weights.num_elements};\n")
        if ir.position_embedding:
            f.write(f"parameter POS_EMB_ADDR  = {ir.position_embedding.weight_base_addr};\n")
            f.write(f"parameter POS_EMB_SIZE  = {ir.position_embedding.weights.num_elements};\n")

        f.write(f"\n// Transformer block addresses\n")
        for block in ir.blocks:
            idx = block.block_idx
            f.write(f"\n// Block {idx}\n")
            if block.attention:
                for proj_name, proj in [("Q", block.attention.q_proj),
                                         ("K", block.attention.k_proj),
                                         ("V", block.attention.v_proj),
                                         ("OUT", block.attention.out_proj)]:
                    if proj:
                        f.write(f"parameter B{idx}_{proj_name}_W_ADDR = {proj.weight_base_addr};\n")
                        f.write(f"parameter B{idx}_{proj_name}_W_SIZE = {proj.weights.num_elements};\n")
                        if proj.bias:
                            f.write(f"parameter B{idx}_{proj_name}_B_ADDR = {proj.bias_base_addr};\n")
            if block.mlp_fc1:
                f.write(f"parameter B{idx}_FC1_W_ADDR = {block.mlp_fc1.weight_base_addr};\n")
                f.write(f"parameter B{idx}_FC1_W_SIZE = {block.mlp_fc1.weights.num_elements};\n")
            if block.mlp_fc2:
                f.write(f"parameter B{idx}_FC2_W_ADDR = {block.mlp_fc2.weight_base_addr};\n")
                f.write(f"parameter B{idx}_FC2_W_SIZE = {block.mlp_fc2.weights.num_elements};\n")

        # LM head
        if ir.lm_head:
            f.write(f"\n// LM Head\n")
            f.write(f"parameter LM_HEAD_W_ADDR = {ir.lm_head.weight_base_addr};\n")
            f.write(f"parameter LM_HEAD_W_SIZE = {ir.lm_head.weights.num_elements};\n")


def _write_weight_rom(ir, filepath: str):
    """Generate a unified weight ROM module that loads all .hex files."""
    with open(filepath, 'w') as f:
        f.write("// ===================================================\n")
        f.write("// fpGPT Compiler -- Auto-generated Unified Weight ROM\n")
        f.write("// Infers Cyclone V M10K block RAM via $readmemh\n")
        f.write("// ===================================================\n\n")

        depth = ir.total_weight_bytes
        addr_width = max(1, (depth - 1).bit_length())

        f.write(f"module weight_rom #(\n")
        f.write(f"    parameter DATA_WIDTH = {ir.bit_width},\n")
        f.write(f"    parameter ADDR_WIDTH = {addr_width},\n")
        f.write(f"    parameter DEPTH      = {depth}\n")
        f.write(f") (\n")
        f.write(f"    input  wire                  clk,\n")
        f.write(f"    input  wire [ADDR_WIDTH-1:0] addr,\n")
        f.write(f"    output reg  [DATA_WIDTH-1:0] data\n")
        f.write(f");\n\n")

        f.write(f"    // M10K block RAM inference (Quartus will map this automatically)\n")
        f.write(f"    reg [DATA_WIDTH-1:0] mem [0:DEPTH-1];\n\n")

        f.write(f"    initial begin\n")
        f.write(f'        $readmemh("weights_unified.hex", mem);\n')
        f.write(f"    end\n\n")

        f.write(f"    always @(posedge clk) begin\n")
        f.write(f"        data <= mem[addr];\n")
        f.write(f"    end\n\n")

        f.write(f"endmodule\n")


def main():
    parser = argparse.ArgumentParser(
        description="fpGPT Compiler: Trained PyTorch model → Synthesizable Verilog"
    )
    parser.add_argument("--model", type=str, required=True,
                        help="Path to trained model checkpoint (.pt)")
    parser.add_argument("--out", type=str, default="build",
                        help="Output directory for generated files")
    parser.add_argument("--precision", type=int, default=8, choices=[8, 16],
                        help="Quantization bit width (8 or 16)")
    parser.add_argument("--fmt", type=str, default="both", choices=["mif", "hex", "both"],
                        help="Weight file format to export")
    args = parser.parse_args()
    compile_model(args)


if __name__ == "__main__":
    main()
