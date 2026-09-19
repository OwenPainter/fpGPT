"""
fpGPT Model — Micro-GPT (Character-Level Transformer Decoder)

A tiny causal transformer designed to fit entirely within the Cyclone V
FPGA's on-chip M10K block RAM (~556 KB) when quantized to INT8.

Default configuration (~210K params, ~210 KB at INT8):
  - Vocabulary:      64 characters (printable ASCII subset)
  - Context length:  64 tokens
  - d_model:         64
  - Num heads:       4  (head_dim = 16)
  - Num layers:      4
  - d_ff (MLP):      256
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────
# Default model configuration
# ─────────────────────────────────────────────────
DEFAULT_CONFIG = {
    "vocab_size": 64,       # Printable ASCII subset (space through '~' minus some)
    "max_seq_len": 64,      # Maximum context window
    "d_model": 64,          # Embedding / hidden dimension
    "num_heads": 4,         # Number of attention heads
    "num_layers": 4,        # Number of transformer blocks
    "d_ff": 256,            # Feed-forward hidden dimension (4x d_model)
    "dropout": 0.1,         # Dropout rate (training only, ignored in hardware)
}


# ─────────────────────────────────────────────────
# Character tokenizer (simple ASCII mapping)
# ─────────────────────────────────────────────────
class CharTokenizer:
    """
    Maps printable ASCII characters to integer token IDs [0, vocab_size).

    Special tokens:
      0 = <pad>
      1 = <sos> (start of sequence)
      2 = <eos> (end of sequence)
      3..vocab_size-1 = printable characters
    """

    def __init__(self, vocab_size: int = 64):
        self.vocab_size = vocab_size
        self.special_tokens = {"<pad>": 0, "<sos>": 1, "<eos>": 2}
        self.num_special = len(self.special_tokens)

        # Build char <-> id mapping for printable ASCII
        printable = [chr(i) for i in range(32, 127)]  # space through '~'
        max_chars = vocab_size - self.num_special
        self.chars = printable[:max_chars]

        self.char_to_id = {c: i + self.num_special for i, c in enumerate(self.chars)}
        self.id_to_char = {i + self.num_special: c for i, c in enumerate(self.chars)}

        # Add special tokens to reverse map
        for tok, idx in self.special_tokens.items():
            self.id_to_char[idx] = tok

    def encode(self, text: str) -> list:
        """Convert a string to a list of token IDs."""
        return [self.char_to_id.get(c, self.special_tokens["<pad>"]) for c in text]

    def decode(self, ids: list) -> str:
        """Convert a list of token IDs back to a string."""
        chars = []
        for i in ids:
            c = self.id_to_char.get(i, "")
            if c not in self.special_tokens:
                chars.append(c)
        return "".join(chars)


# ─────────────────────────────────────────────────
# Model components
# ─────────────────────────────────────────────────
class CausalSelfAttention(nn.Module):
    """
    Multi-head causal (masked) self-attention.

    In hardware, this maps to:
      - 4 linear projections (Q, K, V, Out) using DSP MAC units
      - A causal mask stored as a constant ROM
      - Softmax approximated via lookup table
    """

    def __init__(self, config: dict):
        super().__init__()
        d_model = config["d_model"]
        num_heads = config["num_heads"]
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.d_model = d_model

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.attn_dropout = nn.Dropout(config.get("dropout", 0.1))

        # Causal mask: lower-triangular
        max_len = config["max_seq_len"]
        mask = torch.tril(torch.ones(max_len, max_len))
        self.register_buffer("causal_mask", mask.unsqueeze(0).unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        # Project to Q, K, V
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention with causal mask
        scale = 1.0 / math.sqrt(self.head_dim)
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = attn.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        # Weighted sum and output projection
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)


class MLP(nn.Module):
    """
    Two-layer feed-forward network with GELU activation.

    In hardware, this maps to two sequential dense_layer instances
    with a GELU LUT between them.
    """

    def __init__(self, config: dict):
        super().__init__()
        d_model = config["d_model"]
        d_ff = config["d_ff"]
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(config.get("dropout", 0.1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc2(F.gelu(self.fc1(x))))


class TransformerBlock(nn.Module):
    """
    Pre-norm transformer decoder block.
    LayerNorm -> Attention -> Residual -> LayerNorm -> MLP -> Residual
    """

    def __init__(self, config: dict):
        super().__init__()
        d_model = config["d_model"]
        self.ln1 = nn.LayerNorm(d_model)
        self.attention = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


# ─────────────────────────────────────────────────
# Complete Micro-GPT model
# ─────────────────────────────────────────────────
class MicroGPT(nn.Module):
    """
    Character-level GPT model small enough to be synthesized entirely
    into FPGA on-chip memory as hardwired silicon.

    After training, use the fpGPT compiler to:
      1. Quantize all weights to INT8
      2. Export .mif files for Quartus M10K initialization
      3. Generate Verilog RTL that implements the forward pass
    """

    def __init__(self, config: dict = None):
        super().__init__()
        if config is None:
            config = DEFAULT_CONFIG.copy()
        self.config = config

        vocab_size = config["vocab_size"]
        max_seq_len = config["max_seq_len"]
        d_model = config["d_model"]
        num_layers = config["num_layers"]

        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        self.emb_dropout = nn.Dropout(config.get("dropout", 0.1))

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(num_layers)
        ])

        # Output head
        self.final_ln = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying: share token embedding weights with LM head
        # (saves memory — critical for on-chip budget)
        self.lm_head.weight = self.token_embedding.weight

        # Initialize weights
        self.apply(self._init_weights)
        self._param_count = sum(p.numel() for p in self.parameters())

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    @property
    def param_count(self) -> int:
        return self._param_count

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None):
        """
        Args:
            idx: Token indices, shape (batch, seq_len).
            targets: Target token indices for loss computation, shape (batch, seq_len).

        Returns:
            logits: Shape (batch, seq_len, vocab_size).
            loss: Cross-entropy loss if targets provided, else None.
        """
        B, T = idx.shape
        device = idx.device

        # Token + positional embeddings
        tok_emb = self.token_embedding(idx)                    # (B, T, d_model)
        pos = torch.arange(0, T, dtype=torch.long, device=device)
        pos_emb = self.position_embedding(pos)                 # (T, d_model)
        x = self.emb_dropout(tok_emb + pos_emb)

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        # Output head
        x = self.final_ln(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=0,  # ignore <pad>
            )

        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int = 100,
                 temperature: float = 0.8, top_k: int = None) -> torch.Tensor:
        """
        Autoregressive token generation (greedy / top-k sampling).

        This is the software reference implementation. The hardware equivalent
        is the gpt_controller.v FSM.
        """
        for _ in range(max_new_tokens):
            # Crop to max_seq_len
            idx_cond = idx[:, -self.config["max_seq_len"]:]

            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            probs = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_tok], dim=1)

        return idx

    def summary(self) -> str:
        """Print a summary of the model architecture and memory footprint."""
        int8_bytes = self._param_count
        lines = [
            f"+==========================================+",
            f"|         MicroGPT Architecture            |",
            f"+==========================================+",
            f"|  Vocab size:       {self.config['vocab_size']:>6}               |",
            f"|  Context length:   {self.config['max_seq_len']:>6}               |",
            f"|  d_model:          {self.config['d_model']:>6}               |",
            f"|  Num heads:        {self.config['num_heads']:>6}               |",
            f"|  Num layers:       {self.config['num_layers']:>6}               |",
            f"|  d_ff (MLP):       {self.config['d_ff']:>6}               |",
            f"+------------------------------------------+",
            f"|  Total params:     {self._param_count:>6,}             |",
            f"|  INT8 size:        {int8_bytes / 1024:>6.1f} KB           |",
            f"|  M10K budget:      {int8_bytes / (556 * 1024) * 100:>5.1f}%             |",
            f"+==========================================+",
        ]
        return "\n".join(lines)
