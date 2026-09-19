"""Emit the numeric ABI and configured attention wrappers consumed by RTL."""
import json
from pathlib import Path
from model.fixed_point import gelu_table


def export_fixed(ir, output_dir):
    root = Path(output_dir)
    rtl, weights = root/'rtl', root/'weights'
    rtl.mkdir(parents=True, exist_ok=True)
    weights.mkdir(parents=True, exist_ok=True)
    manifest = {'version': 1, 'data_width': ir.bit_width, 'accumulator_width': 64,
                'scale_rule': 'real = integer * 2**(-fractional_bits)',
                'rounding': {'parameter': 'nearest ties-to-even', 'shift': 'floor', 'division': 'toward zero'},
                'formats': ir.formats, 'linear': {}, 'layer_norm': {}, 'attention': {}, 'gelu': {},
                'unified_rom': {'word_width': 8, 'byte_order': 'little', 'depth': ir.total_weight_bytes}}
    lines = ['// Exact numeric contract. Include inside a module.']
    for name, frac in ir.formats.items():
        lines.append(f'localparam {name.upper()}_FRAC = {frac};')

    def wrapper(name, module, params, declarations, ports):
        (rtl/(name+'.v')).write_text(f'// Generated numeric settings.\nmodule {name}(\n    '+
            ',\n    '.join(declarations)+'\n);\n'+module+' #(\n    '+
            ',\n    '.join(f'.{key}({value})' for key,value in params.items())+') core (\n    '+
            ',\n    '.join(f'.{p}({p})' for p in ports.split())+'\n);\nendmodule\n')

    def residual(name, fa, fb, fout):
        wrapper(name, 'fixed_residual', dict(DATA_WIDTH=ir.bit_width,A_FRAC=fa,B_FRAC=fb,OUT_FRAC=fout),
                [f'input wire signed [{ir.bit_width-1}:0] a,b', f'output wire signed [{ir.bit_width-1}:0] y'], 'a b y')

    residual('embedding_add', ir.formats['tok_emb'], ir.formats['pos_emb'], ir.formats['embeddings'])

    def linear(layer):
        item = {'input_frac': layer.input_frac, 'weight_frac': layer.weights.shift_bits,
                'output_frac': layer.output_frac, 'shift': layer.requant_shift,
                'bias_frac': layer.input_frac+layer.weights.shift_bits, 'bias_width': 64,
                'weight_byte_addr': layer.weight_base_addr,
                'bias_byte_addr': layer.bias_base_addr if layer.bias is not None else None}
        manifest['linear'][layer.name] = item
        lines.append(f'localparam {layer.name.upper()}_SHIFT = {layer.requant_shift};')
        xw, yw, ww = max(1,(layer.in_features-1).bit_length()), max(1,(layer.out_features-1).bit_length()), max(1,(layer.in_features*layer.out_features-1).bit_length())
        wrapper(layer.name+'_fixed', 'fixed_linear', dict(DATA_WIDTH=ir.bit_width, IN_FEATURES=layer.in_features,
                OUT_FEATURES=layer.out_features, SHIFT=layer.requant_shift),
                ['input wire clk,rst_n,start,x_load,w_load,b_load', f'input wire [{xw-1}:0] x_addr',
                 f'input wire [{ww-1}:0] w_addr', f'input wire [{yw-1}:0] b_addr',
                 f'input wire signed [{ir.bit_width-1}:0] x_data,w_data', 'input wire signed [63:0] b_data',
                 'output wire busy,done,y_valid', f'output wire [{yw-1}:0] y_addr',
                 f'output wire signed [{ir.bit_width-1}:0] y_data'],
                'clk rst_n start x_load w_load b_load x_addr w_addr b_addr x_data w_data b_data busy done y_valid y_addr y_data')

    def norm(layer):
        manifest['layer_norm'][layer.name] = {'input_frac': layer.input_frac, 'output_frac': layer.output_frac,
            'gamma_frac': 14, 'gamma_width': 32, 'beta_frac': layer.output_frac, 'beta_width': 32,
            'statistics_extra_frac': 8, 'normalized_frac': 14, 'epsilon_int': layer.epsilon_int,
            'gamma_byte_addr': layer.gamma_base_addr, 'beta_byte_addr': layer.beta_base_addr}
        lines.append(f"localparam signed [63:0] {layer.name.upper()}_EPSILON = 64'sd{layer.epsilon_int};")
        aw = max(1,(layer.normalized_shape-1).bit_length())
        wrapper(layer.name+'_fixed', 'fixed_layer_norm', dict(DATA_WIDTH=ir.bit_width,DIM=layer.normalized_shape,
                OUT_FRAC=layer.output_frac,EPSILON_INT=f"64'sd{layer.epsilon_int}"),
                ['input wire clk,rst_n,load,start', f'input wire [{aw-1}:0] load_addr',
                 f'input wire signed [{ir.bit_width-1}:0] x_data', 'input wire signed [31:0] gamma_data,beta_data',
                 'output wire busy,done,y_valid', f'output wire [{aw-1}:0] y_addr',
                 f'output wire signed [{ir.bit_width-1}:0] y_data'],
                'clk rst_n load start load_addr x_data gamma_data beta_data busy done y_valid y_addr y_data')

    for i, block in enumerate(ir.blocks):
        previous = ir.formats['embeddings' if i==0 else f'block{i-1}_res2']
        residual(f'block{i}_residual1',previous,ir.formats[f'block{i}_out'],ir.formats[f'block{i}_res1'])
        residual(f'block{i}_residual2',ir.formats[f'block{i}_res1'],ir.formats[f'block{i}_fc2'],ir.formats[f'block{i}_res2'])
        a = block.attention
        projections = [a.q_proj, a.k_proj, a.v_proj, a.out_proj]
        for p in projections+[block.mlp_fc1, block.mlp_fc2]:
            linear(p)
        norm(block.ln1)
        norm(block.ln2)
        settings = dict(DATA_WIDTH=ir.bit_width, D_MODEL=ir.d_model, NUM_HEADS=ir.num_heads,
                        MAX_SEQ_LEN=ir.max_seq_len, Q_SHIFT=a.q_proj.requant_shift,
                        K_SHIFT=a.k_proj.requant_shift, V_SHIFT=a.v_proj.requant_shift,
                        OUT_SHIFT=a.out_proj.requant_shift, SCORE_MULT=a.score_mult,
                        SCORE_SHIFT=a.score_shift)
        manifest['attention'][a.name] = settings
        for name, value in settings.items():
            lines.append(f'localparam BLOCK{i}_ATTENTION_{name} = {value};')
        # Host loading images in exactly the attention core's projection order.
        for suffix, values, width in (
                ('weights', [int(v) for p in projections for v in p.weights.data.flat], ir.bit_width),
                ('biases', [int(v) for p in projections for v in (p.bias.data.flat if p.bias else [0]*ir.d_model)], 64)):
            path = weights/f'block{i}_attention_{suffix}.hex'
            path.write_text(''.join(f'{v & ((1<<width)-1):0{width//4}X}\n' for v in values))
        d, t = ir.d_model, ir.max_seq_len
        xw, ww, bw, lw = max(1,(d*t-1).bit_length()), max(1,(4*d*d-1).bit_length()), max(1,(4*d-1).bit_length()), t.bit_length()
        declarations = [
            'input wire clk, rst_n, x_load, w_load, b_load, start',
            f'input wire [{xw-1}:0] x_load_addr', f'input wire [{ww-1}:0] w_load_addr',
            f'input wire [{bw-1}:0] b_load_addr',
            f'input wire signed [{ir.bit_width-1}:0] x_load_data, w_load_data',
            'input wire signed [63:0] b_load_data', f'input wire [{lw-1}:0] seq_len',
            'output wire busy, done, error, y_valid', f'output wire [{xw-1}:0] y_addr',
            f'output wire signed [{ir.bit_width-1}:0] y_data']
        ports = 'clk rst_n x_load w_load b_load start x_load_addr w_load_addr b_load_addr x_load_data w_load_data b_load_data seq_len busy done error y_valid y_addr y_data'.split()
        # The fixed-contract wrappers run one block in isolation, so the
        # KV-cache controls are tied off (full-sequence prefill, layer 0).
        # NUM_LAYERS configures the cache array size but is not part of the
        # frozen manifest.
        li_w = max(1, (ir.num_layers - 1).bit_length())
        core_params = list(settings.items()) + [('NUM_LAYERS', ir.num_layers)]
        connections = [f'.{p}({p})' for p in ports] + [
            f".cache_len({lw}'d0)", f".layer_idx({li_w}'d0)"]
        (rtl/f'attention_block{i}.v').write_text(
            f'// Generated fixed-point settings; host supplies the loading protocol.\nmodule attention_block{i}(\n    '+
            ',\n    '.join(declarations)+'\n);\nattention #(\n    '+
            ',\n    '.join(f'.{k}({v})' for k,v in core_params)+') core (\n    '+
            ',\n    '.join(connections)+'\n);\nendmodule\n')
        fin, fout = ir.formats[f'block{i}_fc1'], ir.formats[f'block{i}_gelu']
        filename = f'block{i}_gelu.hex'
        table = gelu_table(ir.bit_width, fin, fout)
        (weights/filename).write_text(''.join(f'{int(v) & ((1<<ir.bit_width)-1):0{ir.bit_width//4}X}\n' for v in table))
        manifest['gelu'][f'block{i}'] = {'input_frac': fin, 'output_frac': fout, 'file': 'weights/'+filename,
                                     'entries': len(table), 'bytes': len(table)*ir.bit_width//8}
        wrapper(f'block{i}_gelu', 'fixed_gelu', dict(DATA_WIDTH=ir.bit_width,MEM_FILE=f'"weights/{filename}"'),
                ['input wire clk', f'input wire [{ir.bit_width-1}:0] x', f'output wire signed [{ir.bit_width-1}:0] y'], 'clk x y')
    norm(ir.final_ln)
    linear(ir.lm_head)
    (rtl/'fixed_params.vh').write_text('\n'.join(lines)+'\n')
    (root/'fixed_point.json').write_text(json.dumps(manifest, indent=2)+'\n')
    (root/'activation_formats.json').write_text(json.dumps(ir.formats, indent=2)+'\n')
    return manifest
