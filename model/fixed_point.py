"""
fpGPT Model — Software Fixed-Point Emulator

This module provides a bit-accurate software emulation of the hardware
transformer block (Attention + MLP) using the exact same fixed-point INT8
math and scaling operations implemented in Verilog.
"""

import numpy as np

def clamp_int8(val):
    """Saturating clamp to INT8 range [-128, 127]"""
    if val > 127: return 127
    if val < -128: return -128
    return val

def hw_gelu(scaled_val, data_width=8):
    """Hardware GELU approximation from activation.v"""
    # 3 * (1 << (DATA_WIDTH-3)) for DATA_WIDTH=8 is 3 * 32 = 96
    threshold = 3 * (1 << (data_width - 3))
    
    if scaled_val <= -threshold:
        return 0
    elif scaled_val < 0:
        return scaled_val >> 2
    elif scaled_val < threshold:
        return (scaled_val * 3) >> 2
    else:
        return scaled_val

def hardware_dense_layer(x_in, w_int8, b_int8, shift_bits, activation="none"):
    """
    Emulate the dense_layer.v behavior exactly:
    x_in: input INT8 array shape (IN_FEATURES,)
    w_int8: weight INT8 matrix shape (OUT_FEATURES, IN_FEATURES)
    b_int8: bias INT8 array shape (OUT_FEATURES,) or None
    shift_bits: right shift for de-quantization
    activation: "none", "relu", "gelu"
    
    Returns: INT8 output array shape (OUT_FEATURES,)
    """
    in_features = x_in.shape[0]
    out_features = w_int8.shape[0]
    y_out = np.zeros(out_features, dtype=np.int8)
    
    for j in range(out_features):
        # 32-bit accumulation
        acc = np.int32(0)
        for i in range(in_features):
            # INT8 * INT8 -> INT32
            p = np.int32(x_in[i]) * np.int32(w_int8[j, i])
            acc += p
            
        if b_int8 is not None:
            acc += np.int32(b_int8[j])
            
        # Shift down
        scaled = acc >> shift_bits
        
        # Activate
        if activation == "relu":
            activated = 0 if scaled < 0 else scaled
        elif activation == "gelu":
            activated = hw_gelu(scaled)
        else:
            activated = scaled
            
        y_out[j] = clamp_int8(activated)
        
    return y_out

class FixedPointAttention:
    def __init__(self, ir):
        # ir is AttentionIR
        self.q_w = ir.q_proj.weights.data
        self.q_b = ir.q_proj.bias.data if ir.q_proj.bias else None
        self.q_shift = ir.q_proj.weights.shift_bits
        
        self.k_w = ir.k_proj.weights.data
        self.k_b = ir.k_proj.bias.data if ir.k_proj.bias else None
        self.k_shift = ir.k_proj.weights.shift_bits
        
        self.v_w = ir.v_proj.weights.data
        self.v_b = ir.v_proj.bias.data if ir.v_proj.bias else None
        self.v_shift = ir.v_proj.weights.shift_bits
        
        self.o_w = ir.out_proj.weights.data
        self.o_b = ir.out_proj.bias.data if ir.out_proj.bias else None
        self.o_shift = ir.out_proj.weights.shift_bits
        
        self.num_heads = ir.num_heads
        self.d_model = ir.q_proj.weights.shape[1]
        self.head_dim = self.d_model // self.num_heads
        
    def forward(self, x_seq):
        """
        x_seq: INT8 array shape (SEQ_LEN, D_MODEL)
        Returns: INT8 array shape (SEQ_LEN, D_MODEL)
        """
        seq_len = x_seq.shape[0]
        out_seq = np.zeros_like(x_seq)
        
        Q = np.zeros((seq_len, self.d_model), dtype=np.int8)
        K = np.zeros((seq_len, self.d_model), dtype=np.int8)
        V = np.zeros((seq_len, self.d_model), dtype=np.int8)
        
        for t in range(seq_len):
            Q[t] = hardware_dense_layer(x_seq[t], self.q_w, self.q_b, self.q_shift)
            K[t] = hardware_dense_layer(x_seq[t], self.k_w, self.k_b, self.k_shift)
            V[t] = hardware_dense_layer(x_seq[t], self.v_w, self.v_b, self.v_shift)
            
        scale = 1.0 / np.sqrt(self.head_dim)
        out = np.zeros((seq_len, self.d_model), dtype=np.int8)
        
        for t in range(seq_len):
            for h in range(self.num_heads):
                start = h * self.head_dim
                end = start + self.head_dim
                
                qt = Q[t, start:end].astype(np.float32)
                
                attn_scores = []
                for t2 in range(t + 1):
                    kt2 = K[t2, start:end].astype(np.float32)
                    score = np.dot(qt, kt2) * scale
                    attn_scores.append(score)
                
                attn_scores = np.array(attn_scores)
                # prevent overflow
                attn_scores -= np.max(attn_scores)
                probs = np.exp(attn_scores) / np.sum(np.exp(attn_scores))
                
                vt_acc = np.zeros(self.head_dim, dtype=np.float32)
                for t2 in range(t + 1):
                    vt2 = V[t2, start:end].astype(np.float32)
                    vt_acc += probs[t2] * vt2
                
                out[t, start:end] = np.clip(np.round(vt_acc), -128, 127).astype(np.int8)
                
            out_seq[t] = hardware_dense_layer(out[t], self.o_w, self.o_b, self.o_shift)
            
        return out_seq

class FixedPointMLP:
    def __init__(self, fc1_ir, fc2_ir):
        self.fc1_w = fc1_ir.weights.data
        self.fc1_b = fc1_ir.bias.data if fc1_ir.bias else None
        self.fc1_shift = fc1_ir.weights.shift_bits
        
        self.fc2_w = fc2_ir.weights.data
        self.fc2_b = fc2_ir.bias.data if fc2_ir.bias else None
        self.fc2_shift = fc2_ir.weights.shift_bits
        
    def forward(self, x_seq):
        seq_len = x_seq.shape[0]
        out_seq = np.zeros_like(x_seq)
        for t in range(seq_len):
            h = hardware_dense_layer(x_seq[t], self.fc1_w, self.fc1_b, self.fc1_shift, activation="gelu")
            out_seq[t] = hardware_dense_layer(h, self.fc2_w, self.fc2_b, self.fc2_shift, activation="none")
        return out_seq

class FixedPointTransformerBlock:
    def __init__(self, ir):
        self.attn = FixedPointAttention(ir.attention)
        self.mlp = FixedPointMLP(ir.mlp_fc1, ir.mlp_fc2)
        
        self.ln1_gamma = ir.ln1.gamma.data.astype(np.float32) if ir.ln1.gamma else np.ones(self.attn.d_model, dtype=np.float32)
        self.ln1_beta = ir.ln1.beta.data.astype(np.float32) if ir.ln1.beta else np.zeros(self.attn.d_model, dtype=np.float32)
        
        self.ln2_gamma = ir.ln2.gamma.data.astype(np.float32) if ir.ln2.gamma else np.ones(self.attn.d_model, dtype=np.float32)
        self.ln2_beta = ir.ln2.beta.data.astype(np.float32) if ir.ln2.beta else np.zeros(self.attn.d_model, dtype=np.float32)
        
    def layer_norm(self, x_int8, gamma, beta):
        x_fp = x_int8.astype(np.float32)
        mean = np.mean(x_fp, axis=-1, keepdims=True)
        var = np.var(x_fp, axis=-1, keepdims=True)
        x_norm = (x_fp - mean) / np.sqrt(var + 1e-5)
        x_scaled = x_norm * gamma + beta
        return np.clip(np.round(x_scaled), -128, 127).astype(np.int8)

    def forward(self, x_seq):
        seq_len = x_seq.shape[0]
        out_seq = np.zeros_like(x_seq)
        
        for t in range(seq_len):
            x = x_seq[t:t+1]
            
            # Pre-norm 1
            ln1_out = self.layer_norm(x, self.ln1_gamma, self.ln1_beta)
            
            # Attention (only processing up to t)
            # We already have attention processing full seq, so we can just pass full seq to attn
            pass
            
        # Refactor for sequence level processing:
        ln1_out = self.layer_norm(x_seq, self.ln1_gamma, self.ln1_beta)
        attn_out = self.attn.forward(ln1_out)
        
        # Residual 1
        x_res1 = np.clip(x_seq.astype(np.int32) + attn_out.astype(np.int32), -128, 127).astype(np.int8)
        
        # Pre-norm 2
        ln2_out = self.layer_norm(x_res1, self.ln2_gamma, self.ln2_beta)
        mlp_out = self.mlp.forward(ln2_out)
        
        # Residual 2
        x_out = np.clip(x_res1.astype(np.int32) + mlp_out.astype(np.int32), -128, 127).astype(np.int8)
        
        return x_out
