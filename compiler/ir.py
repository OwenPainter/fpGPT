"""
fpGPT Compiler — Intermediate Representation (IR)

Lightweight data structures describing each layer in the neural network
so the Verilog code generator knows exactly what hardware to emit.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional
import numpy as np


class LayerType(Enum):
    """Supported hardware layer types."""
    EMBEDDING = "embedding"
    LINEAR = "linear"
    ATTENTION = "attention"
    LAYER_NORM = "layer_norm"
    RELU = "relu"
    GELU_APPROX = "gelu_approx"
    SOFTMAX = "softmax"
    RESIDUAL_ADD = "residual_add"


class ActivationType(Enum):
    """Activation function applied after a linear layer."""
    NONE = "none"
    RELU = "relu"
    GELU = "gelu"


@dataclass
class QuantizedTensor:
    """
    A quantized weight or bias tensor ready for hardware.

    Attributes:
        data: The quantized integer values (int8 or int16 numpy array).
        scale: The floating-point scale factor used during quantization.
              Exactly 2**(-shift_bits), so real_value = data * scale.
        shift_bits: Fractional bits of the stored tensor (not a MAC output shift).
        shape: Original tensor shape (rows, cols) for matrices.
        bit_width: Number of bits per element (8 or 16).
    """
    data: np.ndarray
    scale: float
    shift_bits: int
    shape: tuple
    bit_width: int = 8

    @property
    def num_elements(self) -> int:
        return int(np.prod(self.shape))

    @property
    def size_bytes(self) -> int:
        return self.num_elements * (self.bit_width // 8)


@dataclass
class LinearLayerIR:
    """IR for a fully-connected (dense / linear) layer."""
    name: str
    in_features: int
    out_features: int
    weights: Optional[QuantizedTensor] = None
    bias: Optional[QuantizedTensor] = None
    activation: ActivationType = ActivationType.NONE
    # Memory layout: base address in the unified weight ROM
    weight_base_addr: int = 0
    bias_base_addr: int = 0
    input_frac: int = 0
    output_frac: int = 0
    requant_shift: int = 0


@dataclass
class EmbeddingLayerIR:
    """IR for a token or position embedding lookup table."""
    name: str
    num_embeddings: int  # vocabulary size or max sequence length
    embedding_dim: int
    weights: Optional[QuantizedTensor] = None
    weight_base_addr: int = 0


@dataclass
class LayerNormIR:
    """IR for layer normalization with learned gamma/beta."""
    name: str
    normalized_shape: int  # = d_model
    gamma: Optional[QuantizedTensor] = None  # scale
    beta: Optional[QuantizedTensor] = None   # shift
    gamma_base_addr: int = 0
    beta_base_addr: int = 0
    input_frac: int = 0
    output_frac: int = 0
    epsilon_int: int = 1


@dataclass
class AttentionIR:
    """
    IR for a multi-head causal self-attention block.

    Contains the Q/K/V projection layers, output projection,
    and attention parameters (num_heads, head_dim).
    """
    name: str
    d_model: int
    num_heads: int
    head_dim: int  # = d_model // num_heads
    q_proj: LinearLayerIR = None
    k_proj: LinearLayerIR = None
    v_proj: LinearLayerIR = None
    out_proj: LinearLayerIR = None
    score_mult: int = 1
    score_shift: int = 0


@dataclass
class TransformerBlockIR:
    """IR for a complete transformer decoder block."""
    name: str
    block_idx: int
    ln1: LayerNormIR = None       # pre-attention layer norm
    attention: AttentionIR = None
    ln2: LayerNormIR = None       # pre-MLP layer norm
    mlp_fc1: LinearLayerIR = None  # MLP first linear (d_model -> d_ff)
    mlp_fc2: LinearLayerIR = None  # MLP second linear (d_ff -> d_model)


@dataclass
class ModelIR:
    """
    Top-level IR for the entire micro-GPT model.

    This is the complete description the Verilog code generator consumes
    to emit synthesizable hardware.
    """
    name: str = "micro_gpt"
    # Architecture hyperparameters
    vocab_size: int = 64
    max_seq_len: int = 64
    d_model: int = 64
    num_heads: int = 4
    num_layers: int = 4
    d_ff: int = 256  # MLP hidden dimension
    bit_width: int = 8

    # Layer descriptions
    token_embedding: Optional[EmbeddingLayerIR] = None
    position_embedding: Optional[EmbeddingLayerIR] = None
    blocks: List[TransformerBlockIR] = field(default_factory=list)
    final_ln: Optional[LayerNormIR] = None
    lm_head: Optional[LinearLayerIR] = None  # output projection to vocab

    # Memory layout summary
    total_params: int = 0
    total_weight_bytes: int = 0
    formats: dict = field(default_factory=dict)

    def compute_memory_layout(self):
        """
        Walk all layers and assign sequential base addresses in the
        unified weight ROM. Returns total memory required in bytes.
        """
        addr = 0

        def assign(tensor: Optional[QuantizedTensor], layer, attr_name: str):
            nonlocal addr
            if tensor is not None:
                if attr_name == "weights":
                    layer.weight_base_addr = addr
                elif attr_name == "bias":
                    layer.bias_base_addr = addr
                elif attr_name == "gamma":
                    layer.gamma_base_addr = addr
                elif attr_name == "beta":
                    layer.beta_base_addr = addr
                addr += tensor.size_bytes

        # Token embedding
        if self.token_embedding:
            assign(self.token_embedding.weights, self.token_embedding, "weights")

        # Position embedding
        if self.position_embedding:
            assign(self.position_embedding.weights, self.position_embedding, "weights")

        # Transformer blocks
        for block in self.blocks:
            # Pre-attention LayerNorm
            if block.ln1:
                assign(block.ln1.gamma, block.ln1, "gamma")
                assign(block.ln1.beta, block.ln1, "beta")

            # Attention Q/K/V/Out projections
            if block.attention:
                for proj in [block.attention.q_proj, block.attention.k_proj,
                             block.attention.v_proj, block.attention.out_proj]:
                    if proj:
                        assign(proj.weights, proj, "weights")
                        assign(proj.bias, proj, "bias")

            # Pre-MLP LayerNorm
            if block.ln2:
                assign(block.ln2.gamma, block.ln2, "gamma")
                assign(block.ln2.beta, block.ln2, "beta")

            # MLP
            if block.mlp_fc1:
                assign(block.mlp_fc1.weights, block.mlp_fc1, "weights")
                assign(block.mlp_fc1.bias, block.mlp_fc1, "bias")
            if block.mlp_fc2:
                assign(block.mlp_fc2.weights, block.mlp_fc2, "weights")
                assign(block.mlp_fc2.bias, block.mlp_fc2, "bias")

        # Final LayerNorm
        if self.final_ln:
            assign(self.final_ln.gamma, self.final_ln, "gamma")
            assign(self.final_ln.beta, self.final_ln, "beta")

        # LM head
        if self.lm_head:
            assign(self.lm_head.weights, self.lm_head, "weights")
            assign(self.lm_head.bias, self.lm_head, "bias")

        self.total_weight_bytes = addr
        return addr

    def summary(self) -> str:
        """Print a human-readable summary of the model IR."""
        lines = [
            f"=== {self.name} Model IR Summary ===",
            f"  Vocab size:       {self.vocab_size}",
            f"  Max sequence len: {self.max_seq_len}",
            f"  d_model:          {self.d_model}",
            f"  Num heads:        {self.num_heads}",
            f"  Num layers:       {self.num_layers}",
            f"  d_ff (MLP):       {self.d_ff}",
            f"  Bit width:        {self.bit_width}-bit",
            f"  Total params:     {self.total_params:,}",
            f"  Weight memory:    {self.total_weight_bytes:,} bytes "
            f"({self.total_weight_bytes / 1024:.1f} KB)",
            f"  M10K budget:      {self.total_weight_bytes / (556 * 1024) * 100:.1f}% "
            f"of 556 KB on-chip",
        ]
        return "\n".join(lines)
