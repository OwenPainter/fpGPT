"""Compare checkpoint fixed-point inference to its PyTorch forward pass.

python -m model.validate_fixed --model checkpoints/micro_gpt.pt --tokens 3,4,5
"""
import argparse
import json
import numpy as np
import torch
from .micro_gpt import MicroGPT
from .fixed_point import FixedMicroGPT
from compiler.quantizer import quantize_model


@torch.no_grad()
def float_trace(model, tokens):
    model.eval()
    idx = torch.tensor([tokens], dtype=torch.long, device=next(model.parameters()).device)
    trace = {}

    def save(name, value):
        trace[name] = value[0].detach().cpu().numpy()
        return value

    x = save('embeddings', model.token_embedding(idx)+model.position_embedding(torch.arange(idx.shape[1], device=idx.device)))
    for i, b in enumerate(model.blocks):
        name = lambda s: f'block{i}_{s}'
        ln = save(name('ln1'), b.ln1(x))
        for suffix in ('q', 'k', 'v'):
            save(name(suffix), getattr(b.attention, suffix+'_proj')(ln))
        a = save(name('out'), b.attention(ln))
        x = save(name('res1'), x+a)
        ln = save(name('ln2'), b.ln2(x))
        hidden = save(name('fc1'), b.mlp.fc1(ln))
        hidden = save(name('gelu'), torch.nn.functional.gelu(hidden))
        mlp = save(name('fc2'), b.mlp.fc2(hidden))
        x = save(name('res2'), x+mlp)
    x = save('final_ln', model.final_ln(x))
    save('logits', model.lm_head(x))
    return trace


def compare(model, ir, tokens):
    trace = float_trace(model, tokens)
    fixed = FixedMicroGPT(ir)
    fixed(tokens)
    report = {}
    for name, integers in fixed.trace.items():
        decoded = integers * 2.0**-ir.formats[name]
        delta = decoded-trace[name]
        limit = 1 << (ir.bit_width-1)
        report[name] = {'rmse': float(np.sqrt(np.mean(delta**2))), 'max_abs_error': float(np.max(np.abs(delta))),
                        'boundary_values': int(np.count_nonzero((integers == -limit) | (integers == limit-1))),
                        'elements': integers.size}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--tokens', required=True, help='comma-separated token IDs')
    parser.add_argument('--precision', type=int, choices=(8,16), default=8)
    parser.add_argument('--formats', help='same activation format JSON as compile.py')
    parser.add_argument('--max-rmse', type=float, help='fail if final-logit RMSE exceeds this threshold')
    args = parser.parse_args()
    checkpoint = torch.load(args.model, map_location='cpu', weights_only=False)
    model = MicroGPT(checkpoint['config'])
    model.load_state_dict(checkpoint['model_state_dict'])
    formats = None
    if args.formats:
        with open(args.formats) as f:
            formats = json.load(f)
    ir = quantize_model(model, model.config, args.precision, formats)
    report = compare(model, ir, [int(v) for v in args.tokens.split(',')])
    print(json.dumps(report, indent=2))
    if args.max_rmse is not None and report['logits']['rmse'] > args.max_rmse:
        raise SystemExit('fixed-point logit error exceeds requested limit')


if __name__ == '__main__':
    main()
