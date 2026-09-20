"""Standalone dense_layer PE-array tests; requires Icarus Verilog.

The testbench models the wide weight ROM directly (it derives the per-lane
addresses from `w_addr`/`w_gather` and returns NUM_PES packed bytes with the
same 1-cycle latency as rom_sync), so dense_layer can be verified without
transformer_engine or the compiler.

Run: python3 -m unittest tests.test_dense_layer -v
"""
import pathlib
import random
import shutil
import subprocess
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _s8(v):
    v &= 0xFF
    return v - 0x100 if v >= 0x80 else v


def reference(x, weights, biases, shift, mode, width=8):
    """Bit-exact model of activation.v applied to the integer MAC sum."""
    limit = 1 << (width - 1)
    result = []
    for j, row in enumerate(weights):
        acc = sum(w * xv for w, xv in zip(row, x))
        if biases is not None:
            acc += biases[j]
        scaled = acc >> shift if shift >= 0 else acc << -shift
        if mode == "relu":
            activated = scaled if scaled > 0 else 0
        elif mode == "gelu":
            threshold = 3 * (1 << (width - 3))
            if scaled <= -threshold:
                activated = 0
            elif scaled < 0:
                activated = scaled >> 2
            elif scaled < threshold:
                activated = (scaled * 3) >> 2
            else:
                activated = scaled
        else:
            activated = scaled
        result.append(min(limit - 1, max(-limit, activated)))
    return result


@unittest.skipUnless(shutil.which('iverilog') and shutil.which('vvp'),
                     'requires Icarus Verilog')
class DenseLayerTests(unittest.TestCase):
    def simulate(self, in_features, out_features, num_pes, seed=1234,
                 has_bias=True, activation="relu", shift=2, width=8,
                 acc_width=32):
        rng = random.Random(seed)
        weights = [[rng.randint(-16, 16) for _ in range(in_features)]
                   for _ in range(out_features)]
        biases = [rng.randint(-24, 24) for _ in range(out_features)] if has_bias else None
        x = [rng.randint(-16, 16) for _ in range(in_features)]
        expected = reference(x, weights, biases, shift, activation, width)

        depth = out_features * in_features + out_features + 8
        w_addr_width = max(1, (depth - 1).bit_length())
        in_aw = max(1, (in_features - 1).bit_length())
        out_aw = max(1, (out_features - 1).bit_length())

        mem_init = []
        for j in range(out_features):
            for i in range(in_features):
                mem_init.append(f"mem[{j*in_features + i}]={weights[j][i] & 0xFF};")
        if has_bias:
            for j in range(out_features):
                mem_init.append(f"mem[{out_features*in_features + j}]={biases[j] & 0xFF};")
        x_init = [f"xmem[{i}]={x[i] & 0xFF};" for i in range(in_features)]
        checks = '\n'.join(f"expected[{j}]={expected[j]};" for j in range(out_features))

        tb = f'''
module tb;
reg clk=0; always #5 clk=~clk;
reg rst_n=0, start=0;
reg [{w_addr_width-1}:0] w_base=0, b_base=0;
wire [{w_addr_width-1}:0] w_addr;
wire w_gather;
wire signed [{num_pes*width-1}:0] w_data;
wire [{in_aw-1}:0] x_addr;
wire signed [{width-1}:0] x_data;
wire y_wr_en;
wire [{out_aw-1}:0] y_addr;
wire signed [{width-1}:0] y_data;

reg [{width-1}:0] mem [0:{depth-1}];
reg [{width-1}:0] xmem [0:{in_features-1}];
integer expected [0:{out_features-1}];
integer got [0:{out_features-1}];
integer kk;
integer written=0;
integer cycles=0;

reg signed [{num_pes*width-1}:0] w_data_r;
reg signed [{width-1}:0] x_data_r;
always @(posedge clk) begin
    for (kk=0; kk<{num_pes}; kk=kk+1)
        if (w_gather) w_data_r[kk*{width} +: {width}] <= mem[w_addr + kk*{in_features}];
        else          w_data_r[kk*{width} +: {width}] <= mem[w_addr + kk];
end
always @(posedge clk) x_data_r <= xmem[x_addr];
assign w_data = w_data_r;
assign x_data = x_data_r;

always @(posedge clk) if (y_wr_en) begin
    got[y_addr] <= y_data;
    written = written + 1;
end
always @(posedge clk) if (start || cycles>0) cycles = cycles + 1;

dense_layer #(
    .DATA_WIDTH({width}), .ACC_WIDTH({acc_width}), .IN_FEATURES({in_features}),
    .OUT_FEATURES({out_features}), .SHIFT_BITS({shift}), .ACTIVATION("{activation}"),
    .W_ADDR_WIDTH({w_addr_width}), .HAS_BIAS({1 if has_bias else 0}),
    .NUM_PES({num_pes})
) dut (
    .clk(clk), .rst_n(rst_n), .start(start), .done(),
    .w_base(w_base), .b_base(b_base),
    .w_addr(w_addr), .w_gather(w_gather), .w_data(w_data),
    .x_addr(x_addr), .x_data(x_data),
    .y_wr_en(y_wr_en), .y_addr(y_addr), .y_data(y_data)
);

integer j;
initial begin
    {chr(10).join(mem_init)}
    {chr(10).join(x_init)}
    {checks}
    w_base = 0;
    b_base = {out_features*in_features};
    repeat(2) @(negedge clk); rst_n=1;
    @(negedge clk); start=1;
    @(negedge clk); start=0;
    wait(written == {out_features}); @(negedge clk); @(negedge clk);
    for (j=0; j<{out_features}; j=j+1)
        if (got[j] !== expected[j])
            $fatal(1, "y[%0d] got %0d expected %0d", j, got[j], expected[j]);
    $display("PASS cycles=%0d written=%0d", cycles, written);
    $finish;
end
initial begin #2000000; $fatal(1, "timeout"); end
endmodule
'''
        with tempfile.TemporaryDirectory(prefix='fpgpt-dense-') as tmp:
            testbench = pathlib.Path(tmp) / 'tb.v'
            testbench.write_text(tb)
            output = pathlib.Path(tmp) / 'sim'
            result = subprocess.run(
                ['iverilog', '-g2012', '-s', 'tb', '-o', str(output),
                 str(ROOT / 'hdl/dense_layer.v'), str(ROOT / 'hdl/mac_unit.v'),
                 str(ROOT / 'hdl/activation.v'), str(testbench)],
                capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            result = subprocess.run(['vvp', str(output)], capture_output=True,
                                    text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn('PASS', result.stdout)
        return expected

    def test_scalar_matches_array(self):
        for num_pes in (1, 2, 4, 8):
            with self.subTest(num_pes=num_pes):
                self.simulate(8, 8, num_pes)

    def test_activation_variants(self):
        for activation in ("relu", "gelu", "none"):
            with self.subTest(activation=activation):
                self.simulate(6, 6, 2, activation=activation, seed=7)

    def test_without_bias(self):
        self.simulate(5, 4, 2, has_bias=False, seed=99)

    def test_wide_layer(self):
        self.simulate(16, 16, 8, seed=2024)

    def test_random_shapes(self):
        rng = random.Random(31337)
        for _ in range(6):
            in_f = rng.choice([3, 5, 7, 8])
            out_f = rng.choice([4, 6, 8, 12])
            pes = rng.choice([1, 2, 3, 4])
            if out_f % pes:
                continue
            with self.subTest(in_f=in_f, out_f=out_f, pes=pes):
                self.simulate(in_f, out_f, pes, seed=rng.randint(0, 1 << 30),
                              has_bias=bool(rng.getrandbits(1)),
                              activation=rng.choice(["relu", "none"]),
                              shift=rng.randint(0, 3))


if __name__ == '__main__':
    unittest.main()
