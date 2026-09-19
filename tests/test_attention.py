"""Integer reference and RTL integration tests; requires Python and Icarus Verilog.

Run: python3 -m unittest discover -s tests -v
"""
import math
import pathlib
import random
import re
import shutil
import subprocess
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def reference(x, weights, biases, heads, shifts, mult, score_shift, width=8):
    """Explicit tensor math, independent of the RTL scheduling/memory layout."""
    dim = len(x[0])
    hd = dim // heads
    limit = 1 << (width - 1)

    def clamp(v):
        return min(limit - 1, max(-limit, v))

    def project(data, p):
        def scaled(value):
            return value >> shifts[p] if shifts[p] >= 0 else value << -shifts[p]
        return [[clamp(scaled(sum(a*b for a, b in zip(token, row)) + bias))
                 for row, bias in zip(weights[p], biases[p])] for token in data]

    q, k, v = [project(x, p) for p in range(3)]
    context = [[0] * dim for _ in x]
    for t in range(len(x)):
        for h in range(heads):
            channels = range(h*hd, (h+1)*hd)
            scores = [(sum(q[t][c]*k[j][c] for c in channels)*mult) >> score_shift
                      for j in range(t+1)]
            maximum = max(scores)
            exps = [round(32768*math.exp(-(maximum-s)/4)) if maximum-s <= 32 else 0
                    for s in scores]
            for c in channels:
                numerator = sum(e*v[j][c] for j, e in enumerate(exps))
                # Verilog signed division truncates toward zero.
                context[t][c] = (1 if numerator >= 0 else -1) * (abs(numerator)//sum(exps))
    return project(context, 3)


@unittest.skipUnless(shutil.which('iverilog') and shutil.which('vvp'), 'requires Icarus Verilog')
class AttentionTests(unittest.TestCase):
    def simulate(self, x, weights, biases, heads=2, shifts=(2, 2, 2, 2),
                 mult=181, score_shift=8, width=8, capacity=None, wrapper_file=None):
        dim = len(x[0])
        # One extra allocated token tests runtime lengths smaller than capacity.
        maximum = capacity or len(x)+1
        expected = reference(x, weights, biases, heads, shifts, mult, score_shift, width)
        stimulus = []
        for p in range(4):
            for r in range(dim):
                stimulus.append(f'@(negedge clk); b_load=1; b_load_addr={p*dim+r}; b_load_data={biases[p][r]};')
                for c in range(dim):
                    stimulus.append(f'@(negedge clk); b_load=0; w_load=1; w_load_addr={p*dim*dim+r*dim+c}; w_load_data={weights[p][r][c]};')
            stimulus.append('@(negedge clk); w_load=0;')
        for t, token in enumerate(x):
            for c, value in enumerate(token):
                stimulus.append(f'@(negedge clk); x_load=1; x_load_addr={t*dim+c}; x_load_data={value};')
        stimulus.append('@(negedge clk); x_load=0;')
        checks = '\n'.join(f'expected[{t*dim+c}]={value};' for t, row in enumerate(expected) for c, value in enumerate(row))
        tb = f'''
module tb;
reg clk=0; always #5 clk=~clk;
reg rst_n=0, start=0, x_load=0, w_load=0, b_load=0;
reg [{max(1, (maximum*dim-1).bit_length())-1}:0] x_load_addr=0;
reg [{max(1, (4*dim*dim-1).bit_length())-1}:0] w_load_addr=0;
reg [{max(1, (4*dim-1).bit_length())-1}:0] b_load_addr=0;
reg signed [{width-1}:0] x_load_data=0, w_load_data=0;
reg signed [63:0] b_load_data=0;
reg [{maximum.bit_length()-1}:0] seq_len=0;
reg [{maximum.bit_length()-1}:0] cache_len=0;
reg [1:0] layer_idx=0;
wire busy, done, error, y_valid;
wire [{max(1, (maximum*dim-1).bit_length())-1}:0] y_addr;
wire signed [{width-1}:0] y_data;
integer expected[0:{len(x)*dim-1}];
integer count=0;
attention #(.DATA_WIDTH({width}), .D_MODEL({dim}), .NUM_HEADS({heads}),
 .MAX_SEQ_LEN({maximum}), .Q_SHIFT({shifts[0]}), .K_SHIFT({shifts[1]}),
 .V_SHIFT({shifts[2]}), .OUT_SHIFT({shifts[3]}),
 .SCORE_MULT({mult}), .SCORE_SHIFT({score_shift})) dut(.*);
always @(negedge clk) if(y_valid) begin
 if(y_addr !== count || y_data !== expected[count])
   $fatal(1, "output %0d address %0d got %0d expected %0d", count, y_addr, y_data, expected[count]);
 count=count+1;
end
initial begin
{checks}
repeat(2) @(negedge clk); rst_n=1;
{chr(10).join(stimulus)}
// Reject a zero length without producing output.
@(negedge clk); start=1; seq_len=0;
@(negedge clk); start=0;
if(!done || !error || busy) $fatal(1,"invalid length protocol");
@(negedge clk);
// Reject lengths greater than capacity when representable by the length port.
if ({maximum} < {2**maximum.bit_length()-1}) begin
 start=1; seq_len={maximum+1};
 @(negedge clk); start=0;
 if(!done || !error || busy) $fatal(1,"oversized length protocol");
 @(negedge clk);
end
// Abort a run with reset; memories must survive and the next run must restart.
start=1; seq_len={len(x)};
@(negedge clk); start=0;
repeat(7) @(negedge clk);
rst_n=0;
@(negedge clk); rst_n=1;
// Two runs ensure all counters, maxima and accumulators restart correctly.
repeat(2) begin
 @(negedge clk); count=0; start=1; seq_len={len(x)};
 @(negedge clk); start=0;
 // Attempts to modify input/weight/bias storage and restart while busy are ignored.
 x_load=1; w_load=1; b_load=1; start=1;
 x_load_addr=0; w_load_addr=0; b_load_addr=0;
 x_load_data=127; w_load_data=127; b_load_data=123456;
 @(negedge clk); x_load=0; w_load=0; b_load=0; start=0;
 wait(done); @(posedge clk);
 if(error || busy || count != {len(x)*dim}) $fatal(1,"completion protocol count=%0d",count);
 @(negedge clk);
 if(done) $fatal(1,"done must pulse");
end
$display("PASS"); $finish;
end
initial begin #10000000; $fatal(1,"timeout"); end
endmodule
'''
        if wrapper_file:
            tb = re.sub(r'attention #\([\s\S]*?\) dut\(\.\*\);',
                        pathlib.Path(wrapper_file).stem+' dut(.*);', tb, count=1)
        with tempfile.TemporaryDirectory(prefix='fpgpt-attention-') as tmp:
            testbench = pathlib.Path(tmp)/'tb.v'
            testbench.write_text(tb)
            output = pathlib.Path(tmp)/'sim'
            result = subprocess.run(['iverilog', '-g2012', '-s', 'tb', '-o', str(output),
                                     str(ROOT/'hdl/attention.v'), str(testbench)]+
                                    ([str(wrapper_file)] if wrapper_file else []), capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            result = subprocess.run(['vvp', str(output)], capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            self.assertIn('PASS', result.stdout)
        return expected

    def test_random_multihead(self):
        rng = random.Random(2714)
        for dim, heads, length, width in [(4, 2, 4, 8), (6, 2, 3, 8), (1, 1, 1, 8), (4, 1, 3, 16)]:
            with self.subTest(dim=dim, heads=heads, width=width):
                x = [[rng.randint(-15, 15) for _ in range(dim)] for _ in range(length)]
                weights = [[[rng.randint(-7, 7) for _ in range(dim)] for _ in range(dim)] for _ in range(4)]
                biases = [[rng.randint(-30, 30) for _ in range(dim)] for _ in range(4)]
                self.simulate(x, weights, biases, heads, (3, 2, 1, 2), width=width)

    def test_uniform_attention_and_causality(self):
        dim = 4
        identity = [[int(r == c) for c in range(dim)] for r in range(dim)]
        zero = [[0]*dim for _ in range(dim)]
        weights = [zero, zero, identity, identity]
        biases = [[0]*dim for _ in range(4)]
        x = [[-10, 4, 20, -6], [30, -8, -10, 4], [127, -128, 127, -128]]
        result = self.simulate(x, weights, biases, shifts=(0, 0, 0, 0))
        self.assertEqual(result[0], x[0])
        self.assertEqual(result[1], [10, -2, 5, -1])
        x[-1] = [-128, 127, -128, 127]
        changed = self.simulate(x, weights, biases, shifts=(0, 0, 0, 0))
        self.assertEqual(result[:2], changed[:2])

    def test_extreme_scores_and_saturation(self):
        identity = [[int(r == c)*127 for c in range(2)] for r in range(2)]
        self.simulate([[127, -128], [-128, 127], [127, 127]], [identity]*4,
                      [[-100, 100]]*4, heads=1, shifts=(0, 0, 0, 0), mult=256)

    def test_default_dimensions(self):
        dim = 64
        identity = [[int(r == c)*64 for c in range(dim)] for r in range(dim)]
        x = [[(i*7)%31-15 for i in range(dim)]]
        self.simulate(x, [identity]*4, [[0]*dim]*4, heads=4,
                      shifts=(7, 7, 7, 7), mult=64, capacity=64)

    def test_full_capacity(self):
        identity = [[1, 0], [0, 1]]
        self.simulate([[1, -2], [3, 4], [-5, 6], [7, -8]], [identity]*4,
                      [[0, 0]]*4, heads=1, shifts=(0, 0, 0, 0), capacity=4)


if __name__ == '__main__':
    unittest.main()
