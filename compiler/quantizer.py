"""
fpGPT Compiler — INT8 Weight Quantizer

Converts floating-point PyTorch model weights into fixed-point INT8 values
suitable for direct hardware implementation in Cyclone V DSP blocks and M10K ROM.

Quantization scheme: Symmetric per-tensor
    q = clamp(round(x / scale), -128, 127)
    scale = 2**(-floor(log2(q_max / max(abs(x)))))
    Weights use exact binary scales; biases use the input-times-weight scale.
"""

import math
import numpy as np
import torch
import torch.nn as nn

from .ir import (
    QuantizedTensor, LinearLayerIR, EmbeddingLayerIR, LayerNormIR,
    AttentionIR, TransformerBlockIR, ModelIR, ActivationType,
)


def quantize_tensor(tensor: torch.Tensor, bit_width: int = 8) -> QuantizedTensor:
    """
    Symmetrically quantize a floating-point tensor to signed integers.

    Args:
        tensor: PyTorch FP32 tensor.
        bit_width: Target bit width (8 or 16).

    Returns:
        QuantizedTensor with int8/int16 data and scale metadata.
    """
    data = tensor.detach().cpu().float().numpy()
    if bit_width not in (8, 16):
        raise ValueError('unsupported tensor width')
    if not np.isfinite(data).all():
        raise ValueError('non-finite model parameter')
    max_val = np.max(np.abs(data))

    if max_val < 1e-10:
        # Near-zero tensor — quantize to all zeros
        q_max = (1 << (bit_width - 1)) - 1  # 127 for 8-bit
        return QuantizedTensor(
            data=np.zeros(data.shape, dtype=np.int8 if bit_width == 8 else np.int16),
            scale=2.0 ** -(bit_width-1),
            shift_bits=bit_width-1,
            shape=data.shape,
            bit_width=bit_width,
        )

    q_max = (1 << (bit_width - 1)) - 1  # 127 for 8-bit, 32767 for 16-bit
    shift_bits = min(24, math.floor(math.log2(q_max / float(max_val))))
    if shift_bits < -16:
        raise ValueError('weight magnitude exceeds supported fixed-point range')
    scale = 2.0 ** -shift_bits

    # Quantize
    q_data = np.clip(np.round(data / scale), -q_max - 1, q_max)
    dtype = np.int8 if bit_width == 8 else np.int16
    q_data = q_data.astype(dtype)


    return QuantizedTensor(
        data=q_data,
        scale=scale,
        shift_bits=shift_bits,
        shape=data.shape,
        bit_width=bit_width,
    )


def quantize_linear(name: str, layer: nn.Linear, bit_width: int = 8,
                     activation: ActivationType = ActivationType.NONE) -> LinearLayerIR:
    """Quantize a PyTorch nn.Linear layer."""
    ir = LinearLayerIR(
        name=name,
        in_features=layer.in_features,
        out_features=layer.out_features,
        weights=quantize_tensor(layer.weight, bit_width),
        activation=activation,
    )
    if layer.bias is not None:
        ir.bias = quantize_tensor(layer.bias, bit_width)
    return ir


def quantize_embedding(name: str, layer: nn.Embedding, bit_width: int = 8) -> EmbeddingLayerIR:
    """Quantize a PyTorch nn.Embedding layer."""
    return EmbeddingLayerIR(
        name=name,
        num_embeddings=layer.num_embeddings,
        embedding_dim=layer.embedding_dim,
        weights=quantize_tensor(layer.weight, bit_width),
    )


def quantize_layer_norm(name: str, layer: nn.LayerNorm, bit_width: int = 8) -> LayerNormIR:
    """Quantize a PyTorch nn.LayerNorm layer (gamma and beta parameters)."""
    ir = LayerNormIR(
        name=name,
        normalized_shape=layer.normalized_shape[0],
    )
    if layer.weight is not None:
        ir.gamma = quantize_tensor(layer.weight, bit_width)
    if layer.bias is not None:
        ir.beta = quantize_tensor(layer.bias, bit_width)
    return ir


def quantize_model(model: nn.Module, config: dict, bit_width: int = 8,
                   formats: dict = None, engine_compatible: bool = False) -> ModelIR:
    """
    Quantize an entire MicroGPT model into the compiler's IR.

    Args:
        model: A trained MicroGPT PyTorch model.
        config: Model config dict with keys: vocab_size, max_seq_len,
                d_model, num_heads, num_layers, d_ff.
        bit_width: Target quantization bit width (8 or 16).
        formats: Optional activation fractional-bit overrides for the fixed
                contract.
        engine_compatible: When True, keep every tensor at ``bit_width`` (the
                uniform byte layout that hdl/transformer_engine.v consumes)
                instead of the mixed-width fixed contract (64-bit biases,
                32-bit LayerNorm parameters). The resulting IR is only for the
                ROM-based engine path.

    Returns:
        A fully populated ModelIR ready for code generation.
    """
    model.eval()

    ir = ModelIR(
        name="micro_gpt",
        vocab_size=config["vocab_size"],
        max_seq_len=config["max_seq_len"],
        d_model=config["d_model"],
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
        d_ff=config["d_ff"],
        bit_width=bit_width,
    )

    # --- Token & Position Embeddings ---
    ir.token_embedding = quantize_embedding(
        "tok_emb", model.token_embedding, bit_width
    )
    ir.position_embedding = quantize_embedding(
        "pos_emb", model.position_embedding, bit_width
    )

    total_params = (ir.token_embedding.weights.num_elements +
                    ir.position_embedding.weights.num_elements)

    # --- Transformer Blocks ---
    for i, block in enumerate(model.blocks):
        block_ir = TransformerBlockIR(name=f"block_{i}", block_idx=i)

        # Pre-attention LayerNorm
        block_ir.ln1 = quantize_layer_norm(f"block_{i}_ln1", block.ln1, bit_width)

        # Multi-Head Attention
        attn = block.attention
        head_dim = config["d_model"] // config["num_heads"]
        attn_ir = AttentionIR(
            name=f"block_{i}_attn",
            d_model=config["d_model"],
            num_heads=config["num_heads"],
            head_dim=head_dim,
            q_proj=quantize_linear(f"block_{i}_q", attn.q_proj, bit_width),
            k_proj=quantize_linear(f"block_{i}_k", attn.k_proj, bit_width),
            v_proj=quantize_linear(f"block_{i}_v", attn.v_proj, bit_width),
            out_proj=quantize_linear(f"block_{i}_out", attn.out_proj, bit_width),
        )
        block_ir.attention = attn_ir

        # Pre-MLP LayerNorm
        block_ir.ln2 = quantize_layer_norm(f"block_{i}_ln2", block.ln2, bit_width)

        # MLP
        block_ir.mlp_fc1 = quantize_linear(
            f"block_{i}_mlp_fc1", block.mlp.fc1, bit_width,
            activation=ActivationType.GELU,
        )
        block_ir.mlp_fc2 = quantize_linear(
            f"block_{i}_mlp_fc2", block.mlp.fc2, bit_width,
        )

        # Count params in this block
        for proj in [attn_ir.q_proj, attn_ir.k_proj, attn_ir.v_proj, attn_ir.out_proj]:
            total_params += proj.weights.num_elements
            if proj.bias:
                total_params += proj.bias.num_elements
        total_params += block_ir.ln1.gamma.num_elements * 2  # gamma + beta
        total_params += block_ir.ln2.gamma.num_elements * 2
        total_params += block_ir.mlp_fc1.weights.num_elements
        if block_ir.mlp_fc1.bias:
            total_params += block_ir.mlp_fc1.bias.num_elements
        total_params += block_ir.mlp_fc2.weights.num_elements
        if block_ir.mlp_fc2.bias:
            total_params += block_ir.mlp_fc2.bias.num_elements

        ir.blocks.append(block_ir)

    # --- Final LayerNorm ---
    ir.final_ln = quantize_layer_norm("final_ln", model.final_ln, bit_width)
    total_params += ir.final_ln.gamma.num_elements * 2

    # --- LM Head ---
    ir.lm_head = quantize_linear("lm_head", model.lm_head, bit_width)
    total_params += ir.lm_head.weights.num_elements
    if ir.lm_head.bias:
        total_params += ir.lm_head.bias.num_elements

    ir.total_params = total_params

    if not engine_compatible:
        # Apply activation formats and accumulator-domain biases before allocation.
        from .fixed_contract import configure
        configure(ir, model, formats)
    # Compute sequential byte layout. Engine-compatible IRs are uniform-width;
    # fixed-contract IRs are mixed-width (see compiler/fixed_contract.py).
    ir.compute_memory_layout()

    return ir
