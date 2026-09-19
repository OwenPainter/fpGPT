"""Select per-stage binary activation scales from representative token sequences."""
import math
import numpy as np


def calibrate_formats(model, sequences, width=8, margin=1.1):
    from model.validate_fixed import float_trace
    if width not in (8,16) or not math.isfinite(margin) or margin < 1:
        raise ValueError('invalid calibration width/margin')
    maxima = {
        'tok_emb': float(model.token_embedding.weight.detach().abs().max()),
        'pos_emb': float(model.position_embedding.weight.detach().abs().max())}
    count = 0
    for tokens in sequences:
        if not tokens or len(tokens) > model.config['max_seq_len'] or any(
                type(v) is not int or not 0 <= v < model.config['vocab_size'] for v in tokens):
            raise ValueError('invalid calibration token sequence')
        count += 1
        for name, data in float_trace(model, tokens).items():
            maximum = float(np.max(np.abs(data)))
            if not math.isfinite(maximum):
                raise ValueError('non-finite calibration activation')
            maxima[name] = max(maxima.get(name, 0), maximum)
    if not count:
        raise ValueError('calibration requires at least one sequence')
    qmax = (1 << (width-1))-1
    return {name: min(15, max(0, math.floor(math.log2(qmax/(maximum*margin))))) if maximum else width-2
            for name, maximum in maxima.items()}
