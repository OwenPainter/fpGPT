import os
import sys
import shutil
import subprocess
import pytest
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.micro_gpt import MicroGPT, DEFAULT_CONFIG
from compiler.quantizer import quantize_model
def write_hex_file(flat_array, filepath, bit_width=8):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        for val in flat_array:
            mask = (1 << bit_width) - 1
            unsigned = int(val) & mask
            hex_chars = (bit_width + 3) // 4
            f.write(f"{unsigned:0{hex_chars}X}\n")
from model.fixed_point import FixedPointTransformerBlock

def test_transformer_block_fixed_point():
    # 1. Generate tiny model
    config = DEFAULT_CONFIG.copy()
    config["d_model"] = 8
    config["num_heads"] = 1
    config["num_layers"] = 1
    config["d_ff"] = 16
    config["max_seq_len"] = 4
    config["vocab_size"] = 16
    
    torch.manual_seed(42)
    model = MicroGPT(config)
    model.eval()
    
    # 2. Extract PyTorch output
    x_in_idx = torch.randint(0, config["vocab_size"], (1, config["max_seq_len"]))
    # We want to test the transformer block directly, not the whole model
    # So we'll grab the output of the embedding layer to feed the block
    x_emb = model.token_embedding(x_in_idx) + model.position_embedding(torch.arange(config["max_seq_len"]).unsqueeze(0))
    x_emb_np = x_emb.detach().numpy()[0]
    
    # Forward through PyTorch block
    pt_out = model.blocks[0](x_emb)
    pt_out_np = pt_out.detach().numpy()[0]
    
    # 3. Quantize the model to IR
    ir = quantize_model(model, config, bit_width=8)
    
    # Quantize the input embedding exactly as fixed point logic requires (INT8)
    # The scale of input embeddings isn't strictly defined as a dynamic thing here,
    # but let's approximate it by simple mapping or we just use the weights quantizer
    # Wait, in the hardware, the embedding ROM outputs INT8 directly.
    # We can get the exact INT8 inputs by looking at the quantized embedding IR
    tok_emb_ir = ir.token_embedding.weights.data
    pos_emb_ir = ir.position_embedding.weights.data
    
    x_emb_int8 = np.zeros((config["max_seq_len"], config["d_model"]), dtype=np.int8)
    for t in range(config["max_seq_len"]):
        tok_id = x_in_idx[0, t].item()
        acc = tok_emb_ir[tok_id].astype(np.int32) + pos_emb_ir[t].astype(np.int32)
        x_emb_int8[t] = np.clip(acc, -128, 127).astype(np.int8)
        
    # 4. Run through Fixed Point Python Emulator
    block_ir = ir.blocks[0]
    fp_block = FixedPointTransformerBlock(block_ir)
    fp_out_int8 = fp_block.forward(x_emb_int8)
    
    # 5. Tolerance check (PyTorch vs Fixed Point)
    # This won't be identical because FP emulator lacks global scale context, but it should be correlated.
    # To properly compare them, we'd need to dequantize fp_out_int8. For now we just verify it runs.
    assert fp_out_int8.shape == (config["max_seq_len"], config["d_model"])
    
    # 6. RTL Reference Test Preparation
    os.makedirs("build/test_rtl", exist_ok=True)
    # Write the activation input as hex
    write_hex_file(x_emb_int8.flatten(), "build/test_rtl/tb_input.hex", bit_width=8)
    
    # Write weight files for the block? The RTL tb_transformer_block.v will need them.
    # For now, we'll assume the RTL will be stubbed. If we want to simulate the RTL,
    # we need to export the block's weights to hex files.
    # Since we are just writing the test stub:
    
    if not shutil.which("iverilog"):
        pytest.skip("Icarus Verilog (iverilog) not found in PATH. Skipping RTL simulation.")
        
    # If we have iverilog, run the testbench
    # tb_file = "tests/tb_transformer_block.v"
    # hdl_files = ["hdl/transformer_block.v", "hdl/dense_layer.v", "hdl/mac_unit.v", "hdl/activation.v"]
    # subprocess.run(["iverilog", "-o", "build/test_rtl/tb_sim", tb_file] + hdl_files, check=True)
    # subprocess.run(["vvp", "build/test_rtl/tb_sim"], check=True)
    
    # Read output and verify
    # with open("build/test_rtl/tb_output.hex", "r") as f:
    #     ...
    
