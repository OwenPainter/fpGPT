import json
import math
from pathlib import Path
import random
import shutil
import subprocess
import sys
import tempfile
import unittest
import numpy as np
import torch
from compiler.quantizer import quantize_model, quantize_tensor
from compiler.fixed_export import export_fixed
from compiler.calibration import calibrate_formats
from compiler.mif_writer import export_all_weights
from model.micro_gpt import MicroGPT
from model.fixed_point import FixedMicroGPT, attention, layer_norm, residual, gelu_table, shift, clip
from model.validate_fixed import compare
import test_attention

ROOT = Path(__file__).resolve().parents[1]


def tiny(width=8):
    torch.manual_seed(2714)
    model = MicroGPT(dict(vocab_size=12, max_seq_len=4, d_model=8, num_heads=2,
                          num_layers=2, d_ff=16, dropout=0)).eval()
    # Exercise nonzero learned biases and nontrivial affine LayerNorm parameters.
    with torch.no_grad():
        for name, p in model.named_parameters():
            if 'bias' in name:
                p.uniform_(-0.02, 0.02)
    return model, quantize_model(model, model.config, width)


class NumericTests(unittest.TestCase):
    def test_compile_and_validate_cli(self):
        model,_ = tiny()
        optimizer = torch.optim.AdamW(model.parameters(),lr=0.001)
        _,loss = model(torch.tensor([[3,4,5]]),targets=torch.tensor([[4,5,6]]))
        loss.backward(); optimizer.step()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            torch.save({'config':model.config,'model_state_dict':model.state_dict()},root/'model.pt')
            (root/'calibration.json').write_text(json.dumps([[3,4,5],[6,7,8]]))
            result = subprocess.run([sys.executable,str(ROOT/'compile.py'),'--model',str(root/'model.pt'),
                                     '--out',str(root/'build'),'--calibration',str(root/'calibration.json')],
                                    capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            result = subprocess.run([sys.executable,'-m','model.validate_fixed','--model',str(root/'model.pt'),
                                     '--tokens','3,4,5','--formats',str(root/'build/activation_formats.json'),
                                     '--max-rmse','0.05'],cwd=ROOT,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            self.assertLess(json.loads(result.stdout)['logits']['rmse'],0.05)

    def test_calibration(self):
        torch.manual_seed(2714)
        model = MicroGPT().eval()
        sequences = [[3,4,5,6], [7,8,9,10]]
        for width,tolerance in ((8,0.025),(16,0.01)):
            formats = calibrate_formats(model,sequences,width)
            ir = quantize_model(model,model.config,width,formats)
            self.assertLess(compare(model,ir,[3,4,5,6])['logits']['rmse'],tolerance)
            self.assertLess(compare(model,ir,[11,12,13,14])['logits']['rmse'],tolerance*2)
        with self.assertRaises(ValueError):
            calibrate_formats(model,[])
    def test_exact_binary_scales(self):
        for width in (8,16):
            for data in ([0.,0.], [-0.2,0.03,0.1], [100.,-50.]):
                q = quantize_tensor(torch.tensor(data), width)
                self.assertEqual(q.scale, 2.0**-q.shift_bits)
                np.testing.assert_array_less(np.abs(q.data*q.scale-data), q.scale/2+1e-6)
        with self.assertRaises(ValueError):
            quantize_tensor(torch.tensor([float('nan')]))

    def test_bias_and_formats(self):
        model, _ = tiny()
        ir = quantize_model(model, model.config, formats={'block0_q': 7, 'block0_out': 4, 'block0_res1': 6})
        p = ir.blocks[0].attention.q_proj
        self.assertEqual(p.bias.bit_width, 64)
        self.assertEqual(p.bias.scale, 2.0**-(p.input_frac+p.weights.shift_bits))
        self.assertEqual(p.requant_shift, p.input_frac+p.weights.shift_bits-p.output_frac)
        self.assertEqual(ir.blocks[0].ln2.input_frac, 6)
        with self.assertRaises(ValueError):
            quantize_model(model, model.config, formats={'typo': 4})

    def test_full_model_against_pytorch(self):
        for width, tolerance in ((8, 0.06), (16, 0.002)):
            model, ir = tiny(width)
            report = compare(model, ir, [3,4,5,6])
            self.assertLess(report['logits']['rmse'], tolerance, report)
            self.assertLess(report['final_ln']['rmse'], 0.5 if width == 8 else 0.02, report)
            fixed = FixedMicroGPT(ir)
            a = fixed([3,4,5,6]); b = fixed([3,4,1,2])
            np.testing.assert_array_equal(a[:2], b[:2])
            with self.assertRaises(ValueError):
                fixed([999])

    def test_residual_single_saturation(self):
        np.testing.assert_array_equal(residual([127,-128], 0, [-127,127], 0, 7, 8), [0,-128])
        np.testing.assert_array_equal(residual([-1,1], 2, [1,-1], 3, 1, 8), [-1,0])

    def test_export_memory_and_wrappers(self):
        from compile import _write_weight_rom
        model, ir = tiny(16)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            export_all_weights(ir, root/'weights', fmt='mif')
            manifest = export_fixed(ir, root)
            _write_weight_rom(ir, root/'rtl/weight_rom.v')
            image = bytes(int(x,16) for x in (root/'weights/weights_unified.hex').read_text().split())
            self.assertEqual(len(image), ir.total_weight_bytes)
            p = ir.blocks[0].attention.q_proj
            addr = p.bias_base_addr
            self.assertEqual(int.from_bytes(image[addr:addr+8], 'little', signed=True), int(p.bias.data[0]))
            self.assertEqual(manifest['attention']['block_0_attn']['Q_SHIFT'], p.requant_shift)
            if shutil.which('iverilog'):
                subprocess.run(['iverilog','-g2012','-s','attention_block0','-o',str(root/'sim'),
                                str(root/'rtl/attention_block0.v'),str(ROOT/'hdl/attention.v')],check=True)
                subprocess.run(['iverilog','-g2012','-o',str(root/'all')]+
                               [str(p) for p in (root/'rtl').glob('*.v')]+
                               [str(ROOT/'hdl'/name) for name in ('attention.v','fixed_linear.v',
                                'fixed_layer_norm.v','fixed_residual.v','fixed_gelu.v')],check=True)


    def test_engine_image_and_board_params(self):
        from compile import _write_board_params
        model, ir = tiny(8)
        ir_engine = quantize_model(model, model.config, 8, engine_compatible=True)
        # The engine path keeps every tensor at the model bit width.
        for t in (ir_engine.token_embedding.weights,
                  ir_engine.position_embedding.weights,
                  ir_engine.blocks[0].attention.q_proj.weights,
                  ir_engine.blocks[0].attention.q_proj.bias,
                  ir_engine.blocks[0].ln1.gamma,
                  ir_engine.blocks[0].ln1.beta,
                  ir_engine.lm_head.weights):
            self.assertEqual(t.bit_width, 8)
        self.assertEqual(ir_engine.total_weight_bytes, ir_engine.total_params)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            export_all_weights(ir_engine, root/'weights/engine', fmt='hex', unified_only=True)
            image = bytes(int(x, 16) for x in
                          (root/'weights/engine/weights_unified.hex').read_text().split())
            self.assertEqual(len(image), ir_engine.total_weight_bytes)
            self.assertEqual(image[0], int(ir_engine.token_embedding.weights.data.flat[0]) & 0xFF)
            _write_board_params(ir, ir_engine, root/'board_params.vh')
            text = (root/'board_params.vh').read_text()
            self.assertIn(f'`define FPGPT_ROM_DEPTH       {ir_engine.total_weight_bytes}', text)
            self.assertIn(f'`define FPGPT_D_MODEL         {model.config["d_model"]}', text)
            self.assertIn(f'`define FPGPT_VOCAB_SIZE      {model.config["vocab_size"]}', text)
            self.assertIn('`define FPGPT_Q_SHIFT', text)
            self.assertIn('`ifndef FPGPT_BOARD_PARAMS_VH', text)


@unittest.skipUnless(shutil.which('iverilog') and shutil.which('vvp'), 'requires Icarus Verilog')
class RtlNumericTests(unittest.TestCase):
    def test_linear_rtl(self):
        for width in (8,16):
            model,ir = tiny(width)
            layer = ir.blocks[0].mlp_fc1
            row = [-13,12,-9,7,0,1,2,3]
            for amount in (layer.requant_shift,-2):
                expected = [clip(shift(sum(x*int(w) for x,w in zip(row,weights))+int(bias),amount),width)
                            for weights,bias in zip(layer.weights.data,layer.bias.data)]
                setup = []
                for i,x in enumerate(row):
                    setup.append(f'@(negedge clk); x_load=1; x_addr={i}; x_data={x};')
                setup.append('@(negedge clk); x_load=0;')
                for i,w in enumerate(layer.weights.data.flat):
                    setup.append(f'@(negedge clk); w_load=1; w_addr={i}; w_data={int(w)};')
                setup.append('@(negedge clk); w_load=0;')
                for i,b in enumerate(layer.bias.data):
                    setup.append(f'@(negedge clk); b_load=1; b_addr={i}; b_data={int(b)};')
                setup.append('@(negedge clk); b_load=0; start=1; @(negedge clk); start=0;')
                checks = '\n'.join(f'expected[{i}]={v};' for i,v in enumerate(expected))
                self.run_tb(f'''module tb;
reg clk=0; always #5 clk=~clk;
reg rst_n=0,start=0,x_load=0,w_load=0,b_load=0;
reg [2:0] x_addr=0; reg [6:0] w_addr=0; reg [3:0] b_addr=0;
reg signed [{width-1}:0] x_data=0,w_data=0; reg signed [63:0] b_data=0;
wire busy,done,y_valid; wire [3:0] y_addr; wire signed [{width-1}:0] y_data;
integer expected[0:15]; integer count=0;
fixed_linear #(.DATA_WIDTH({width}),.IN_FEATURES(8),.OUT_FEATURES(16),.SHIFT({amount})) dut(.*);
always @(negedge clk) if(y_valid) begin
 if(y_addr !== count || y_data !== expected[count]) $fatal(1,"linear %0d got %0d expected %0d",count,y_data,expected[count]);
 count=count+1;
end
initial begin
{checks}
repeat(2) @(negedge clk); rst_n=1;
{chr(10).join(setup)}
wait(done); @(posedge clk); if(count!=16) $fatal(1,"missing output"); $finish;
end
initial begin #100000; $fatal(1,"timeout"); end
endmodule''',['fixed_linear.v'])

    def run_tb(self, body, files, cwd=None):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path/'tb.v').write_text(body)
            compile_result = subprocess.run(['iverilog','-g2012','-s','tb','-o',str(path/'sim'),str(path/'tb.v')]+
                                           [str(ROOT/'hdl'/f) for f in files],capture_output=True,text=True)
            self.assertEqual(compile_result.returncode,0,compile_result.stderr)
            result = subprocess.run(['vvp',str(path/'sim')],cwd=cwd,capture_output=True,text=True,timeout=30)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)

    def test_compiler_attention_vs_rtl_and_pytorch(self):
        for width in (8,16):
            model, ir = tiny(width)
            a = ir.blocks[0].attention
            rng = np.random.default_rng(14)
            x = rng.integers(-2**(a.q_proj.input_frac), 2**a.q_proj.input_frac, size=(3,8))
            projections = [a.q_proj,a.k_proj,a.v_proj,a.out_proj]
            expected = attention(x,a,width)
            harness = test_attention.AttentionTests()
            with tempfile.TemporaryDirectory() as tmp:
                export_fixed(ir,tmp)
                result = harness.simulate(x.tolist(),[p.weights.data.tolist() for p in projections],
                                          [p.bias.data.tolist() for p in projections], heads=2,
                                          shifts=tuple(p.requant_shift for p in projections),
                                          mult=a.score_mult,score_shift=a.score_shift,width=width,
                                          wrapper_file=Path(tmp)/'rtl/attention_block0.v')
            np.testing.assert_array_equal(result,expected)
            with torch.no_grad():
                fp = model.blocks[0].attention(torch.tensor(x*2.0**-a.q_proj.input_frac,dtype=torch.float32)[None])[0].numpy()
            self.assertLess(np.sqrt(np.mean((expected*2.0**-a.out_proj.output_frac-fp)**2)), 0.04 if width==8 else 0.001)

    def test_layernorm_rtl_and_pytorch(self):
        for width in (8,16):
            model, ir = tiny(width)
            ln = ir.blocks[0].ln1
            rng = random.Random(8)
            rows = [[0]*8, [-3]*8, [0,0,0,0,0,0,0,1], [rng.randint(-100,100) for _ in range(8)],
                    [-(1<<(width-1)),(1<<(width-1))-1]*4]
            for row in rows:
                expected = layer_norm([row],ln,width)[0]
                setup = '\n'.join(f'@(negedge clk); load=1; load_addr={i}; x_data={x}; gamma_data={int(g)}; beta_data={int(b)};'
                                  for i,(x,g,b) in enumerate(zip(row,ln.gamma.data,ln.beta.data)))
                checks = '\n'.join(f'expected[{i}]={v};' for i,v in enumerate(expected))
                self.run_tb(f'''module tb;
reg clk=0; always #5 clk=~clk;
reg rst_n=0, load=0, start=0; reg [2:0] load_addr=0;
reg signed [{width-1}:0] x_data=0; reg signed [31:0] gamma_data=0,beta_data=0;
wire busy,done,y_valid; wire [2:0] y_addr; wire signed [{width-1}:0] y_data;
integer expected[0:7]; integer count=0;
fixed_layer_norm #(.DATA_WIDTH({width}),.DIM(8),.OUT_FRAC({ln.output_frac}),.EPSILON_INT(64'sd{ln.epsilon_int})) dut(.*);
always @(negedge clk) if(y_valid) begin
 if(y_addr !== count || y_data !== expected[count]) $fatal(1,"LN %0d got %0d expected %0d",count,y_data,expected[count]);
 count=count+1;
end
initial begin
{checks}
repeat(2) @(negedge clk); rst_n=1;
{setup}
@(negedge clk); load=0; start=1;
@(negedge clk); start=0;
wait(done); @(posedge clk); if(count!=8) $fatal(1,"missing LN output"); $finish;
end
initial begin #10000; $fatal(1,"timeout"); end
endmodule''', ['fixed_layer_norm.v'])
                with torch.no_grad():
                    fp = model.blocks[0].ln1(torch.tensor([row],dtype=torch.float32)*2.0**-ln.input_frac)[0].numpy()
                self.assertLess(np.max(np.abs(expected*2.0**-ln.output_frac-fp)), 0.08 if width==8 else 0.005)

    def test_residual_rtl(self):
        rng = random.Random(5)
        for fa,fb,fo in ((5,3,4),(0,0,7),(7,6,0)):
            pairs = [(rng.randint(-128,127),rng.randint(-128,127)) for _ in range(100)] + [(127,-127),(-128,127)]
            commands = []
            for a,b in pairs:
                e = residual([a],fa,[b],fb,fo,8)[0]
                commands.append(f'a={a}; b={b}; #1; if(y !== $signed(8\'d{int(e)&255})) $fatal(1,"residual");')
            self.run_tb('module tb; reg signed [7:0] a,b; wire signed [7:0] y;'+
                        f'fixed_residual #(.A_FRAC({fa}),.B_FRAC({fb}),.OUT_FRAC({fo})) dut(.*);'+
                        'initial begin '+ '\n'.join(commands)+' $finish; end endmodule',['fixed_residual.v'])

    def test_gelu_rtl(self):
        _, ir = tiny()
        with tempfile.TemporaryDirectory() as tmp:
            export_fixed(ir,tmp)
            values = gelu_table(8,ir.formats['block0_fc1'],ir.formats['block0_gelu'])
            commands = [f'@(negedge clk); x=8\'d{i}; @(negedge clk); if(y !== $signed(8\'d{int(v)&255})) $fatal(1,"GELU");'
                        for i,v in enumerate(values)]
            self.run_tb('module tb; reg clk=0; always #5 clk=~clk; reg [7:0] x=0; wire signed [7:0] y;'+
                        'fixed_gelu #(.MEM_FILE("weights/block0_gelu.hex")) dut(.*); initial begin '+
                        '\n'.join(commands)+' $finish; end endmodule',['fixed_gelu.v'],cwd=tmp)
