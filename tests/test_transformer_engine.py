"""End-to-end integer reference and RTL integration test for transformer_engine.

Builds a tiny random Micro-GPT in the compiler's sequential weight layout,
runs the full forward pass in Icarus Verilog, and compares the vocabulary
projection (logits) of the last token against an independent Python integer
model that mirrors the RTL fixed-point arithmetic.

Run: python3 -m unittest discover -s tests -v
"""
import math
import pathlib
import random
import shutil
import subprocess
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]

WIDTH = 8
DATA_LO = -(1 << (WIDTH - 1))
DATA_HI = (1 << (WIDTH - 1)) - 1


def sat(v):
    return min(DATA_HI, max(DATA_LO, v))


def trunc_div(a, b):
    q = abs(a) // abs(b)
    return q if (a >= 0) == (b >= 0) else -q


def layer_norm(x, gamma, beta, shift):
    n = len(x)
    mean = trunc_div(sum(x), n)
    d = [xi - mean for xi in x]
    variance = trunc_div(sum(di * di for di in d), n)
    root = math.isqrt(variance)
    inv = (1 << shift) if root == 0 else trunc_div(1 << shift, root)
    out = []
    for i in range(n):
        norm = (d[i] * inv) >> shift
        out.append(sat(((norm * gamma[i]) >> shift) + beta[i]))
    return out


def gelu_approx(scaled):
    limit = 3 * (1 << (WIDTH - 3))
    if scaled <= -limit:
        return 0
    if scaled < 0:
        return scaled >> 2
    if scaled < limit:
        return (scaled * 3) >> 2
    return scaled


def dense(x, weights, bias, shift, activation):
    out = []
    for row, b in zip(weights, bias):
        acc = sum(w * xi for w, xi in zip(row, x)) + b
        scaled = acc >> shift
        if activation == "gelu":
            scaled = gelu_approx(scaled)
        out.append(sat(scaled))
    return out


def attention(x, weights, biases, heads, shifts, mult, score_shift):
    """Bit-exact mirror of hdl/attention.v, adapted from tests/test_attention.py."""
    dim = len(x[0])
    hd = dim // heads

    def project(data, p):
        return [[sat((sum(a * b for a, b in zip(token, row)) + bias) >> shifts[p])
                 for row, bias in zip(weights[p], biases[p])] for token in data]

    q, k, v = [project(x, p) for p in range(3)]
    context = [[0] * dim for _ in x]
    for t in range(len(x)):
        for h in range(heads):
            channels = range(h * hd, (h + 1) * hd)
            scores = [(sum(q[t][c] * k[j][c] for c in channels) * mult) >> score_shift
                      for j in range(t + 1)]
            maximum = max(scores)
            exps = [round(32768 * math.exp(-(maximum - s) / 4)) if maximum - s <= 32 else 0
                    for s in scores]
            for c in channels:
                numerator = sum(e * v[j][c] for j, e in enumerate(exps))
                context[t][c] = trunc_div(numerator, sum(exps))
    return project(context, 3)


class GPTControllerTests(unittest.TestCase):
    def build_model(self, rng, dim, heads, length, layers, d_ff, vocab, shifts):
        model = {
            "tok_emb": [[rng.randint(-12, 12) for _ in range(dim)] for _ in range(vocab)],
            "pos_emb": [[rng.randint(-12, 12) for _ in range(dim)] for _ in range(length)],
            "blocks": [],
            "final_g": [rng.randint(64, 127) for _ in range(dim)],
            "final_b": [rng.randint(-8, 8) for _ in range(dim)],
            "lm_w": [[rng.randint(-2, 2) for _ in range(dim)] for _ in range(vocab)],
        }
        for _ in range(layers):
            block = {
                "ln1_g": [rng.randint(64, 127) for _ in range(dim)],
                "ln1_b": [rng.randint(-8, 8) for _ in range(dim)],
                "attn_w": [[[rng.randint(-2, 2) for _ in range(dim)] for _ in range(dim)]
                           for _ in range(4)],
                "attn_b": [[rng.randint(-5, 5) for _ in range(dim)] for _ in range(4)],
                "ln2_g": [rng.randint(64, 127) for _ in range(dim)],
                "ln2_b": [rng.randint(-8, 8) for _ in range(dim)],
                "fc1_w": [[rng.randint(-2, 2) for _ in range(dim)] for _ in range(d_ff)],
                "fc1_b": [rng.randint(-5, 5) for _ in range(d_ff)],
                "fc2_w": [[rng.randint(-2, 2) for _ in range(d_ff)] for _ in range(dim)],
                "fc2_b": [rng.randint(-5, 5) for _ in range(dim)],
            }
            model["blocks"].append(block)
        return model

    def reference(self, model, tokens, dim, heads, d_ff, shifts, mult, score_shift, ln_shift):
        length = len(tokens)
        act = [[sat(model["tok_emb"][tokens[t]][c] + model["pos_emb"][t][c])
                for c in range(dim)] for t in range(length)]
        for block in model["blocks"]:
            ln1 = [layer_norm(act[t], block["ln1_g"], block["ln1_b"], ln_shift)
                   for t in range(length)]
            attn = attention(ln1, block["attn_w"], block["attn_b"], heads,
                             shifts, mult, score_shift)
            act = [[sat(act[t][c] + attn[t][c]) for c in range(dim)] for t in range(length)]
            ln2 = [layer_norm(act[t], block["ln2_g"], block["ln2_b"], ln_shift)
                   for t in range(length)]
            for t in range(length):
                hidden = dense(ln2[t], block["fc1_w"], block["fc1_b"], 0, "gelu")
                mlp = dense(hidden, block["fc2_w"], block["fc2_b"], 0, "none")
                act[t] = [sat(act[t][c] + mlp[c]) for c in range(dim)]
        final = layer_norm(act[-1], model["final_g"], model["final_b"], ln_shift)
        return dense(final, model["lm_w"], [0] * len(model["lm_w"]), 0, "none")

    def rom_image(self, model, dim, length, vocab, d_ff):
        image = []
        image += [v for row in model["tok_emb"] for v in row]
        image += [v for row in model["pos_emb"] for v in row]
        for block in model["blocks"]:
            image += block["ln1_g"] + block["ln1_b"]
            for p in range(4):
                image += [v for row in block["attn_w"][p] for v in row]
                image += block["attn_b"][p]
            image += block["ln2_g"] + block["ln2_b"]
            image += [v for row in block["fc1_w"] for v in row] + block["fc1_b"]
            image += [v for row in block["fc2_w"] for v in row] + block["fc2_b"]
        image += model["final_g"] + model["final_b"]
        image += [v for row in model["lm_w"] for v in row]
        return image

    def simulate(self, model, tokens, dim, heads, length, layers, d_ff, vocab,
                 shifts=(0, 0, 0, 0), mult=64, score_shift=8, ln_shift=7):
        expected = self.reference(model, tokens, dim, heads, d_ff, shifts, mult,
                                  score_shift, ln_shift)
        image = self.rom_image(model, dim, length, vocab, d_ff)
        w_addr_width = max(1, (len(image) - 1).bit_length())
        len_w = max(1, (length + 1 - 1).bit_length()) if length + 1 > 1 else 1
        tok_aw = max(1, (length - 1).bit_length()) if length > 1 else 1
        voc_aw = max(1, (vocab - 1).bit_length()) if vocab > 1 else 1

        hex_path = "weights.hex"
        hex_lines = "\n".join(f"{v & 0xFF:02X}" for v in image)
        token_load = "\n".join(
            f"@(negedge clk); tok_load=1; tok_addr={t}; tok_data={tok};" for t, tok in enumerate(tokens)
        )
        expected_checks = "\n".join(
            f"expected[{i}] = {v};" for i, v in enumerate(expected)
        )

        tb = f'''
module tb;
reg clk=0; always #5 clk=~clk;
reg rst_n=0, start=0, tok_load=0;
reg [{len_w-1}:0] seq_len=0;
reg [{tok_aw-1}:0] tok_addr=0;
reg [{voc_aw-1}:0] tok_data=0;
wire busy, done, error, logits_valid;
wire [{voc_aw-1}:0] logits_addr;
wire signed [7:0] logits_data;
wire [{w_addr_width-1}:0] rom_addr;
reg signed [7:0] rom_data;
reg [7:0] mem [0:{len(image)-1}];
integer expected [0:{len(expected)-1}];
integer captured [0:{vocab-1}];
integer i;

always @(posedge clk) rom_data <= mem[rom_addr];
always @(posedge clk) if (logits_valid) captured[logits_addr] <= logits_data;

transformer_engine #(
    .DATA_WIDTH(8), .ACC_WIDTH(32), .D_MODEL({dim}), .NUM_HEADS({heads}),
    .MAX_SEQ_LEN({length}), .NUM_LAYERS({layers}), .D_FF({d_ff}),
    .VOCAB_SIZE({vocab}),
    .Q_SHIFT({shifts[0]}), .K_SHIFT({shifts[1]}), .V_SHIFT({shifts[2]}),
    .OUT_SHIFT({shifts[3]}), .SCORE_MULT({mult}), .SCORE_SHIFT({score_shift}),
    .LN_SHIFT({ln_shift}), .FC1_SHIFT(0), .FC2_SHIFT(0), .LM_SHIFT(0),
    .W_ADDR_WIDTH({w_addr_width})
) dut(.*);

initial begin
    $readmemh("{hex_path}", mem);
    {expected_checks}
    for (i=0;i<{vocab};i=i+1) captured[i]=999;
    repeat(2) @(negedge clk); rst_n=1;
    {token_load}
    @(negedge clk); tok_load=0; start=1; seq_len={len(tokens)};
    @(negedge clk); start=0;
    wait(done); @(posedge clk);
    if (error || busy) $fatal(1, "controller error/busy");
    for (i=0;i<{vocab};i=i+1)
        if (captured[i] !== expected[i])
            $fatal(1, "logit %0d got %0d expected %0d", i, captured[i], expected[i]);
    $display("PASS");
    $finish;
end
initial begin #20000000; $fatal(1,"timeout"); end
endmodule
'''
        with tempfile.TemporaryDirectory(prefix="fpgpt-controller-") as tmp:
            tmp = pathlib.Path(tmp)
            (tmp / hex_path).write_text(hex_lines + "\n")
            tb_path = tmp / "tb.v"
            tb_path.write_text(tb)
            output = tmp / "sim"
            files = ["transformer_engine.v", "attention.v", "layer_norm.v", "dense_layer.v",
                     "mac_unit.v", "activation.v"]
            result = subprocess.run(
                ["iverilog", "-g2012", "-s", "tb", "-o", str(output)]
                + [str(ROOT / "hdl" / f) for f in files] + [str(tb_path)],
                capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            result = subprocess.run(["vvp", str(output)], capture_output=True, text=True,
                                    timeout=120, cwd=tmp)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("PASS", result.stdout)

    @unittest.skipUnless(shutil.which("iverilog") and shutil.which("vvp"),
                         "requires Icarus Verilog")
    def test_forward_pass(self):
        rng = random.Random(0xF9)
        tokens = [1, 2, 3]
        model = self.build_model(rng, dim=4, heads=2, length=3, layers=2, d_ff=4, vocab=4,
                                 shifts=(0, 0, 0, 0))
        self.simulate(model, tokens, dim=4, heads=2, length=3, layers=2, d_ff=4, vocab=4)

    @unittest.skipUnless(shutil.which("iverilog") and shutil.which("vvp"),
                         "requires Icarus Verilog")
    def test_partial_sequence_and_single_layer(self):
        # Context capacity 4, but only 2 tokens are run.
        rng = random.Random(0x51)
        tokens = [2, 3]
        model = self.build_model(rng, dim=4, heads=1, length=4, layers=1, d_ff=4, vocab=4,
                                 shifts=(0, 0, 0, 0))
        self.simulate(model, tokens, dim=4, heads=1, length=4, layers=1, d_ff=4, vocab=4)

    @unittest.skipUnless(shutil.which("iverilog") and shutil.which("vvp"),
                         "requires Icarus Verilog")
    def test_alternate_head_dimension(self):
        # Non-power-of-two head dimension (d_model=6, heads=2 -> head_dim=3).
        rng = random.Random(0x77)
        tokens = [0, 4, 1]
        model = self.build_model(rng, dim=6, heads=2, length=3, layers=1, d_ff=6, vocab=5,
                                 shifts=(0, 0, 0, 0))
        self.simulate(model, tokens, dim=6, heads=2, length=3, layers=1, d_ff=6, vocab=5)


if __name__ == "__main__":
    unittest.main()
