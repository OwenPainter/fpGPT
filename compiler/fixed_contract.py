"""Binary scale selection and hardware arithmetic contracts.

Stored signed integer q with fractional bits f means q * 2**(-f).
All runtime operations use integer arithmetic; parameter conversion is offline.
"""
import math
import numpy as np
from .ir import QuantizedTensor


def tensor_at(tensor, frac, width):
    data = tensor.detach().cpu().double().numpy()
    scaled = np.rint(np.ldexp(data, frac))  # offline ties-to-even rounding
    if not np.isfinite(scaled).all() or np.any(scaled >= 2**(width-1)) or np.any(scaled < -2**(width-1)):
        raise ValueError(f'parameter does not fit signed {width} bits at fraction {frac}')
    return QuantizedTensor(scaled.astype(f'int{width}'), 2.0**-frac, frac, data.shape, width)


def configure(ir, model, overrides=None):
    if ir.bit_width not in (8, 16):
        raise ValueError('model precision must be 8 or 16')
    if overrides is not None and not isinstance(overrides, dict):
        raise ValueError('activation formats must be a JSON object')
    # Conservative defaults; deployment should provide measured per-stage ranges.
    residual = ir.bit_width-3
    normalized = ir.bit_width-4
    formats = {'tok_emb': residual, 'pos_emb': residual, 'embeddings': residual,
               'final_ln': normalized, 'logits': residual}
    for i in range(ir.num_layers):
        for suffix in ('ln1', 'ln2'):
            formats[f'block{i}_{suffix}'] = normalized
        for suffix in ('q', 'k', 'v', 'out', 'res1', 'res2', 'fc1', 'gelu', 'fc2'):
            formats[f'block{i}_{suffix}'] = residual
    if overrides:
        unknown = set(overrides)-set(formats)
        if unknown:
            raise ValueError(f'unknown activation formats: {sorted(unknown)}')
        formats.update(overrides)
    if any(type(f) is not int or not 0 <= f <= 15 for f in formats.values()):
        raise ValueError('activation fractional bits must be integers in 0..15')
    ir.formats = formats

    def linear(layer, source, fin, fout):
        layer.input_frac, layer.output_frac = fin, fout
        product_frac = fin + layer.weights.shift_bits
        layer.requant_shift = product_frac-fout
        if not -31 <= layer.requant_shift <= 62:
            raise ValueError(f'{layer.name}: unsupported rescale shift')
        layer.bias = tensor_at(source.bias, product_frac, 64) if source.bias is not None else None
        bound = layer.in_features * 2**(2*(ir.bit_width-1))
        if layer.bias is not None:
            bound += max(abs(int(v)) for v in layer.bias.data.flat)
        if bound * 2**max(0, -layer.requant_shift) >= 2**63:
            raise ValueError(f'{layer.name}: possible accumulator/rescale overflow')

    def norm(layer, source, fin, fout):
        layer.input_frac, layer.output_frac = fin, fout
        # Mean and variance retain eight extra fractional bits.
        layer.epsilon_int = max(1, round(source.eps * 2**(2*(fin+8))))
        layer.gamma = tensor_at(source.weight, 14, 32)
        layer.beta = tensor_at(source.bias, fout, 32)
        dim = layer.normalized_shape
        if dim*2**(2*(ir.bit_width+8)) + layer.epsilon_int >= 2**63:
            raise ValueError(f'{layer.name}: LayerNorm statistics may overflow')
        # Conservative normalization/affine bound, including output rescale.
        gamma_max = max(abs(int(v)) for v in layer.gamma.data.flat)
        if math.ceil(math.sqrt(dim))*2**14*gamma_max >= 2**62:
            raise ValueError(f'{layer.name}: LayerNorm affine may overflow')

    ir.token_embedding.weights = tensor_at(model.token_embedding.weight, formats['tok_emb'], ir.bit_width)
    ir.position_embedding.weights = tensor_at(model.position_embedding.weight, formats['pos_emb'], ir.bit_width)
    current = formats['embeddings']
    for i, (block, source) in enumerate(zip(ir.blocks, model.blocks)):
        f = lambda suffix: formats[f'block{i}_{suffix}']
        norm(block.ln1, source.ln1, current, f('ln1'))
        for suffix in ('q', 'k', 'v'):
            linear(getattr(block.attention, suffix+'_proj'), getattr(source.attention, suffix+'_proj'), f('ln1'), f(suffix))
        a = block.attention
        linear(a.out_proj, source.attention.out_proj, f('v'), f('out'))
        # Integer multiplier includes 1/sqrt(head_dim) and quarter-unit score scale.
        a.score_mult = round(2**20 / math.sqrt(a.head_dim))
        a.score_shift = 20 + f('q') + f('k') - 2
        if a.head_dim*2**(2*(ir.bit_width-1))*a.score_mult >= 2**63:
            raise ValueError('attention score multiplication may overflow')
        norm(block.ln2, source.ln2, f('res1'), f('ln2'))
        linear(block.mlp_fc1, source.mlp.fc1, f('ln2'), f('fc1'))
        linear(block.mlp_fc2, source.mlp.fc2, f('gelu'), f('fc2'))
        current = f('res2')
    norm(ir.final_ln, model.final_ln, current, formats['final_ln'])
    linear(ir.lm_head, model.lm_head, formats['final_ln'], formats['logits'])
