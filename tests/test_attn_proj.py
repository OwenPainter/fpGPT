"""Standalone attn_proj tests: parallel attention projections vs a reference.

Instantiates four real hdl/weight_cache.v instances as the per-projection packed
weight ROMs and hdl/attn_proj.v, runs each projection over a token range, and
compares the streamed writes against explicit integer math.

Run: python3 -m unittest tests.test_attn_proj -v
"""
import pathlib
import random
import shutil
import subprocess
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _clamp(v, width=8):
    limit = 1 << (width - 1)
    return min(limit - 1, max(-limit, v))


def reference(weights, biases, xs, shifts, width=8):
    """y[p][t][r] = clamp((sum_c W[p][r][c]*x[p][t][c] + b[p][r]) >> shift[p])."""
    out = []
    for p in range(4):
        proj = []
        for t, x in enumerate(xs[p]):
            row_out = []
            for r, wrow in enumerate(weights[p]):
                acc = sum(w * xv for w, xv in zip(wrow, x)) + biases[p][r]
                row_out.append(_clamp(acc >> shifts[p], width))
            proj.append(row_out)
        out.append(proj)
    return out


@unittest.skipUnless(shutil.which('iverilog') and shutil.which('vvp'),
                     'requires Icarus Verilog')
class AttnProjTests(unittest.TestCase):
    def simulate(self, dim, num_pes, length, max_seq=8, seed=4321,
                 shifts=(2, 3, 1, 2), width=8):
        rng = random.Random(seed)
        weights = [[[rng.randint(-16, 16) for _ in range(dim)]
                    for _ in range(dim)] for _ in range(4)]
        biases = [[rng.randint(-24, 24) for _ in range(dim)] for _ in range(4)]
        xs = [[[rng.randint(-16, 16) for _ in range(dim)] for _ in range(length)]
              for _ in range(4)]
        expected = reference(weights, biases, xs, shifts, width)

        x_depth = max_seq * dim
        w_addr_width = max(1, (dim * dim - 1).bit_length())
        x_aw = max(1, (x_depth - 1).bit_length())
        len_w = max(1, max_seq.bit_length())

        init = []
        for p in range(4):
            for r in range(dim):
                for c in range(dim):
                    init.append(f"wc_w[{p*dim*dim + r*dim + c}]={weights[p][r][c] & 0xFF};")
        for p in range(4):
            for r in range(dim):
                init.append(f"wc_b[{p*dim+r}]={biases[p][r] & 0xFF};")
        for p in range(4):
            for t in range(length):
                for c in range(dim):
                    init.append(f"xall[{p*x_depth + t*dim + c}]={xs[p][t][c] & 0xFF};")
        checks = []
        for p in range(4):
            for t in range(length):
                for r in range(dim):
                    checks.append(
                        f"expected[{p*x_depth + t*dim + r}]={expected[p][t][r]};")

        tb = f'''
module tb;
reg clk=0; always #5 clk=~clk;
reg rst_n=0;
reg start=0;
reg [1:0] projection=0;
reg [{len_w-1}:0] length={length};
reg [{len_w-1}:0] cache_len=0;

wire [{w_addr_width-1}:0] w_addr_q, w_addr_k, w_addr_v, w_addr_o;
wire w_gather_q, w_gather_k, w_gather_v, w_gather_o;
wire signed [{num_pes*width-1}:0] w_data_q, w_data_k, w_data_v, w_data_o;
wire [{x_aw-1}:0] x_addr;
wire signed [{width-1}:0] x_data;
wire wr_en;
wire [{x_aw-1}:0] wr_addr;
wire signed [{width-1}:0] wr_data;
wire busy, done;

reg we_w=0, we_b=0;
reg [1:0] fill_p=0;
reg [{w_addr_width-1}:0] waddr=0, baddr=0;
reg signed [{width-1}:0] wdata=0, bdata=0;
reg [{width-1}:0] wc_w [0:{4*dim*dim-1}];
reg [{width-1}:0] wc_b [0:{4*dim-1}];
wire we_w_q = we_w && (fill_p==0), we_w_k = we_w && (fill_p==1);
wire we_w_v = we_w && (fill_p==2), we_w_o = we_w && (fill_p==3);
wire we_b_q = we_b && (fill_p==0), we_b_k = we_b && (fill_p==1);
wire we_b_v = we_b && (fill_p==2), we_b_o = we_b && (fill_p==3);

weight_cache #(.DATA_WIDTH({width}), .NUM_PES({num_pes}), .IN_FEATURES({dim}),
 .OUT_ROWS({dim}), .ADDR_WIDTH({w_addr_width})) wc_q (
 .clk(clk), .we_w(we_w_q), .waddr(waddr), .wdata(wdata),
 .we_b(we_b_q), .baddr(baddr), .bdata(bdata),
 .raddr(w_addr_q), .gather(w_gather_q), .rdata(w_data_q));
weight_cache #(.DATA_WIDTH({width}), .NUM_PES({num_pes}), .IN_FEATURES({dim}),
 .OUT_ROWS({dim}), .ADDR_WIDTH({w_addr_width})) wc_k (
 .clk(clk), .we_w(we_w_k), .waddr(waddr), .wdata(wdata),
 .we_b(we_b_k), .baddr(baddr), .bdata(bdata),
 .raddr(w_addr_k), .gather(w_gather_k), .rdata(w_data_k));
weight_cache #(.DATA_WIDTH({width}), .NUM_PES({num_pes}), .IN_FEATURES({dim}),
 .OUT_ROWS({dim}), .ADDR_WIDTH({w_addr_width})) wc_v (
 .clk(clk), .we_w(we_w_v), .waddr(waddr), .wdata(wdata),
 .we_b(we_b_v), .baddr(baddr), .bdata(bdata),
 .raddr(w_addr_v), .gather(w_gather_v), .rdata(w_data_v));
weight_cache #(.DATA_WIDTH({width}), .NUM_PES({num_pes}), .IN_FEATURES({dim}),
 .OUT_ROWS({dim}), .ADDR_WIDTH({w_addr_width})) wc_o (
 .clk(clk), .we_w(we_w_o), .waddr(waddr), .wdata(wdata),
 .we_b(we_b_o), .baddr(baddr), .bdata(bdata),
 .raddr(w_addr_o), .gather(w_gather_o), .rdata(w_data_o));

reg signed [{width-1}:0] xall [0:{4*x_depth-1}];
reg signed [{width-1}:0] xmem [0:{x_depth-1}];
reg signed [{width-1}:0] x_data_r;
always @(posedge clk) x_data_r <= xmem[x_addr];
assign x_data = x_data_r;

attn_proj #(.DATA_WIDTH({width}), .ACC_WIDTH(32), .D_MODEL({dim}),
 .NUM_PES({num_pes}), .MAX_SEQ_LEN({max_seq}),
 .Q_SHIFT({shifts[0]}), .K_SHIFT({shifts[1]}), .V_SHIFT({shifts[2]}),
 .OUT_SHIFT({shifts[3]}), .W_ADDR_WIDTH({w_addr_width}),
 .X_AW({x_aw}), .LEN_WIDTH({len_w})) dut (
 .clk(clk), .rst_n(rst_n), .start(start), .projection(projection),
 .length(length), .cache_len(cache_len),
 .w_addr_q(w_addr_q), .w_addr_k(w_addr_k), .w_addr_v(w_addr_v), .w_addr_o(w_addr_o),
 .w_gather_q(w_gather_q), .w_gather_k(w_gather_k), .w_gather_v(w_gather_v), .w_gather_o(w_gather_o),
 .w_data_q(w_data_q), .w_data_k(w_data_k), .w_data_v(w_data_v), .w_data_o(w_data_o),
 .x_addr(x_addr), .x_data(x_data),
 .wr_en(wr_en), .wr_addr(wr_addr), .wr_data(wr_data),
 .busy(busy), .done(done));

integer got [0:{4*x_depth-1}];
integer expected [0:{4*x_depth-1}];
integer cur_proj=0;
integer p, t, r, f, i;
always @(posedge clk) if (wr_en) got[cur_proj*{x_depth} + wr_addr] = wr_data;

initial begin
    {chr(10).join(init)}
    {chr(10).join(checks)}
    repeat(2) @(negedge clk); rst_n=1;
    for (p=0; p<4; p=p+1) begin
        fill_p = p;
        for (f=0; f<{dim*dim}; f=f+1) begin
            @(negedge clk); we_w=1; waddr=f; wdata=wc_w[p*{dim*dim}+f];
        end
        @(negedge clk); we_w=0;
        for (f=0; f<{dim}; f=f+1) begin
            @(negedge clk); we_b=1; baddr=f; bdata=wc_b[p*{dim}+f];
        end
        @(negedge clk); we_b=0;
    end

    for (p=0; p<4; p=p+1) begin
        cur_proj = p;
        projection = p;
        for (i=0; i<{x_depth}; i=i+1) xmem[i] = xall[p*{x_depth}+i];
        @(negedge clk); start=1;
        @(negedge clk); start=0;
        wait(done); @(negedge clk);
        for (t=0; t<{length}; t=t+1)
            for (r=0; r<{dim}; r=r+1)
                if (got[p*{x_depth} + t*{dim} + r] !== expected[p*{x_depth} + t*{dim} + r])
                    $fatal(1, "p=%0d t=%0d r=%0d got %0d exp %0d",
                           p, t, r, got[p*{x_depth}+t*{dim}+r],
                           expected[p*{x_depth}+t*{dim}+r]);
    end
    $display("PASS");
    $finish;
end
initial begin #5000000; $fatal(1, "timeout"); end
endmodule
'''
        with tempfile.TemporaryDirectory(prefix='fpgpt-attnproj-') as tmp:
            tb_path = pathlib.Path(tmp) / 'tb.v'
            tb_path.write_text(tb)
            output = pathlib.Path(tmp) / 'sim'
            files = [ROOT / 'hdl/dense_layer.v', ROOT / 'hdl/mac_unit.v',
                     ROOT / 'hdl/activation.v', ROOT / 'hdl/weight_cache.v',
                     ROOT / 'hdl/attn_proj.v', tb_path]
            result = subprocess.run(
                ['iverilog', '-g2012', '-s', 'tb', '-o', str(output)] +
                [str(f) for f in files], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            result = subprocess.run(['vvp', str(output)], capture_output=True,
                                    text=True, timeout=120)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn('PASS', result.stdout)

    def test_scalar_and_packed(self):
        for num_pes in (1, 2, 4):
            with self.subTest(num_pes=num_pes):
                self.simulate(8, num_pes, length=4)

    def test_saturating(self):
        self.simulate(4, 2, length=3, shifts=(0, 0, 0, 0), seed=11)

    def test_shift_variants(self):
        self.simulate(4, 4, length=3, shifts=(0, 1, 3, 5), seed=77)


if __name__ == '__main__':
    unittest.main()
