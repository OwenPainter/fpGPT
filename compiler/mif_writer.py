"""
fpGPT Compiler — Memory Initialization File (MIF) & HEX Writer

Generates two output formats for loading quantized weights into FPGA memory:
  1. Intel Quartus .mif format — for altsyncram / M10K block inference
  2. Verilog $readmemh .hex format — for simulation testbenches

Both formats pack INT8 weights row-major into addressable memory words.
"""

import os
import numpy as np
from .ir import QuantizedTensor, ModelIR


def _to_hex(val: int, width_bits: int) -> str:
    """
    Convert a signed integer to an unsigned hex string of the given bit width.
    E.g., -1 with 8-bit -> 'FF', 42 with 8-bit -> '2A'
    """
    mask = (1 << width_bits) - 1
    unsigned = val & mask
    hex_chars = (width_bits + 3) // 4  # number of hex digits
    return f"{unsigned:0{hex_chars}X}"


def write_mif(tensor: QuantizedTensor, filepath: str, word_width: int = 8):
    """
    Write a QuantizedTensor to Intel MIF format.

    Args:
        tensor: The quantized weight tensor to serialize.
        filepath: Output .mif file path.
        word_width: Bits per memory word (default 8 for INT8).
    """
    flat = tensor.data.flatten()
    depth = len(flat)

    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with open(filepath, 'w') as f:
        f.write(f"-- fpGPT Compiler: Auto-generated weight memory\n")
        f.write(f"-- Tensor shape: {tensor.shape}, Scale: {tensor.scale:.6e}\n")
        f.write(f"-- Shift bits: {tensor.shift_bits}\n\n")
        f.write(f"WIDTH={word_width};\n")
        f.write(f"DEPTH={depth};\n\n")
        f.write(f"ADDRESS_RADIX=HEX;\n")
        f.write(f"DATA_RADIX=HEX;\n\n")
        f.write(f"CONTENT BEGIN\n")

        for addr, val in enumerate(flat):
            hex_val = _to_hex(int(val), word_width)
            hex_addr = f"{addr:04X}"
            f.write(f"  {hex_addr} : {hex_val};\n")

        f.write(f"END;\n")


def write_hex(tensor: QuantizedTensor, filepath: str, word_width: int = 8):
    """
    Write a QuantizedTensor to Verilog $readmemh format.

    Each line is one hex value, addresses are implicit (sequential from 0).
    """
    flat = tensor.data.flatten()
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with open(filepath, 'w') as f:
        f.write(f"// fpGPT Compiler: Auto-generated weight memory\n")
        f.write(f"// Shape: {tensor.shape}, Scale: {tensor.scale:.6e}, "
                f"Shift: {tensor.shift_bits}\n")
        for val in flat:
            f.write(_to_hex(int(val), word_width) + "\n")


def export_all_weights(ir: ModelIR, output_dir: str, fmt: str = "both"):
    """
    Export all quantized weights from a ModelIR to .mif and/or .hex files.

    Creates one file per weight tensor, organized in subdirectories:
        output_dir/
          tok_emb_weights.mif
          pos_emb_weights.mif
          block_0_q_weights.mif
          block_0_q_bias.mif
          ...
          lm_head_weights.mif

    Args:
        ir: Fully quantized ModelIR.
        output_dir: Base directory for output files.
        fmt: "mif", "hex", or "both".

    Returns:
        List of (name, filepath, size_bytes) tuples for all exported files.
    """
    os.makedirs(output_dir, exist_ok=True)
    exported = []

    def _export(name: str, tensor: QuantizedTensor):
        if tensor is None:
            return
        if fmt in ("mif", "both"):
            path = os.path.join(output_dir, f"{name}.mif")
            write_mif(tensor, path, tensor.bit_width)
            exported.append((name, path, tensor.size_bytes))
        if fmt in ("hex", "both"):
            path = os.path.join(output_dir, f"{name}.hex")
            write_hex(tensor, path, tensor.bit_width)
            exported.append((name, path, tensor.size_bytes))

    # Token & position embeddings
    if ir.token_embedding:
        _export("tok_emb_weights", ir.token_embedding.weights)
    if ir.position_embedding:
        _export("pos_emb_weights", ir.position_embedding.weights)

    # Transformer blocks
    for block in ir.blocks:
        idx = block.block_idx
        # LayerNorm 1
        if block.ln1:
            _export(f"block{idx}_ln1_gamma", block.ln1.gamma)
            _export(f"block{idx}_ln1_beta", block.ln1.beta)
        # Attention projections
        if block.attention:
            for proj_name, proj in [("q", block.attention.q_proj),
                                     ("k", block.attention.k_proj),
                                     ("v", block.attention.v_proj),
                                     ("out", block.attention.out_proj)]:
                if proj:
                    _export(f"block{idx}_{proj_name}_weights", proj.weights)
                    if proj.bias:
                        _export(f"block{idx}_{proj_name}_bias", proj.bias)
        # LayerNorm 2
        if block.ln2:
            _export(f"block{idx}_ln2_gamma", block.ln2.gamma)
            _export(f"block{idx}_ln2_beta", block.ln2.beta)
        # MLP
        if block.mlp_fc1:
            _export(f"block{idx}_mlp_fc1_weights", block.mlp_fc1.weights)
            if block.mlp_fc1.bias:
                _export(f"block{idx}_mlp_fc1_bias", block.mlp_fc1.bias)
        if block.mlp_fc2:
            _export(f"block{idx}_mlp_fc2_weights", block.mlp_fc2.weights)
            if block.mlp_fc2.bias:
                _export(f"block{idx}_mlp_fc2_bias", block.mlp_fc2.bias)

    # Final LayerNorm
    if ir.final_ln:
        _export("final_ln_gamma", ir.final_ln.gamma)
        _export("final_ln_beta", ir.final_ln.beta)

    # LM Head
    if ir.lm_head:
        _export("lm_head_weights", ir.lm_head.weights)
        if ir.lm_head.bias:
            _export("lm_head_bias", ir.lm_head.bias)

    return exported
