"""Throughput harness for the board generation controller.

Runs generation_controller.v in Icarus Verilog for a fixed prompt and
GEN_TOKENS decode steps, counts the clock cycles of the generation burst, and
reports cycles/token and tokens/second at the 50 MHz and 150 MHz clock targets.

Run: python3 -m unittest tests.test_throughput -v

Set FPGPT_THROUGHPUT_DEFAULT=1 to additionally measure a default-sized model
(d_model=64, 4 layers, d_ff=256); that run is much slower in simulation.

Packed weight bus (Task 2): hdl/rom_sync.v and hdl/weight_cache.v can return
NUM_PES packed weight bytes per read, and the engine stops re-streaming the
unified image every layer/token once hdl/dense_layer.v exposes its NUM_PES
packed port (Task 1). This harness logs the single-byte baseline today; the
packed path is measured once that port and the generation_controller wiring
land. Baseline for the default shape: ~1.03M cycles for a 3-token prefill /
~437k cycles per generated token.
"""
import os
import pathlib
import random
import shutil
import subprocess
import tempfile
import unittest

from tests.test_transformer_engine import GPTControllerTests

ROOT = pathlib.Path(__file__).resolve().parents[1]

# Clock targets for the board (50 MHz oscillator, 150 MHz PLL output).
CLOCK_HZ = {"50 MHz": 50_000_000, "150 MHz": 150_000_000}


class ThroughputTests(unittest.TestCase):
    def measure(self, model, tokens, dim, heads, length, layers, d_ff, vocab,
                gen_tokens, shifts=(0, 0, 0, 0), mult=64, score_shift=8,
                ln_shift=7, num_pes=1, timeout=180):
        """Return (cycles, gen_tokens) for one generation burst."""
        helper = GPTControllerTests()
        image = helper.rom_image(model, dim, length, vocab, d_ff)
        w_addr_width = max(1, (len(image) - 1).bit_length())
        tok_aw = max(1, (length - 1).bit_length()) if length > 1 else 1

        # Prompt tokens must be >= 3 so the byte mapping round-trips.
        assert all(t >= 3 for t in tokens), "prompt tokens must map to characters"
        prompt_bytes = [t - 3 + 32 for t in tokens]
        prompt_send = "\n".join(
            f"@(negedge clk); token_in={b}; token_valid=1;\n@(negedge clk); token_valid=0;"
            for b in prompt_bytes
        )
        hex_lines = "\n".join(f"{v & 0xFF:02X}" for v in image)

        tb = f'''
module tb;
reg clk=0; always #10 clk=~clk;
reg rst_n=0;
reg [7:0] token_in=0;
reg token_valid=0;
wire [7:0] token_out;
wire token_out_valid, busy;
wire [31:0] gen_cycles;
integer cycles=0;
integer tokens=0;

generation_controller #(
    .DATA_WIDTH(8), .ACC_WIDTH(32), .D_MODEL({dim}), .NUM_HEADS({heads}),
    .MAX_SEQ_LEN({length}), .NUM_LAYERS({layers}), .D_FF({d_ff}),
    .VOCAB_SIZE({vocab}),
    .Q_SHIFT({shifts[0]}), .K_SHIFT({shifts[1]}), .V_SHIFT({shifts[2]}),
    .OUT_SHIFT({shifts[3]}), .SCORE_MULT({mult}), .SCORE_SHIFT({score_shift}),
    .LN_SHIFT({ln_shift}), .FC1_SHIFT(0), .FC2_SHIFT(0), .LM_SHIFT(0),
    .W_ADDR_WIDTH({w_addr_width}), .ROM_DEPTH({len(image)}),
    .ROM_MEM_FILE("weights.hex"), .GEN_TOKENS({gen_tokens}), .NUM_PES({num_pes})
) dut (
    .clk(clk), .rst_n(rst_n), .token_in(token_in), .token_valid(token_valid),
    .token_out(token_out), .token_out_valid(token_out_valid), .busy(busy),
    .gen_cycles(gen_cycles)
);

always @(posedge clk) if (busy) cycles = cycles + 1;
always @(posedge clk) if (token_out_valid) begin
    tokens = tokens + 1;
    if (tokens == {gen_tokens}) begin
        $display("THROUGHPUT cycles=%0d tokens=%0d gen_cycles=%0d",
                 cycles, tokens, gen_cycles);
        $finish;
    end
end

initial begin
    repeat(2) @(negedge clk); rst_n=1;
    {prompt_send}
    @(negedge clk); token_in=8'h0A; token_valid=1;
    @(negedge clk); token_valid=0;
end
initial begin #{timeout * 1_000_000}; $fatal(1,"throughput timeout"); end
endmodule
'''
        with tempfile.TemporaryDirectory(prefix="fpgpt-throughput-") as tmp:
            tmp = pathlib.Path(tmp)
            (tmp / "weights.hex").write_text(hex_lines + "\n")
            tb_path = tmp / "tb.v"
            tb_path.write_text(tb)
            output = tmp / "sim"
            files = ["generation_controller.v", "transformer_engine.v", "attention.v",
                     "layer_norm.v", "dense_layer.v", "mac_unit.v", "activation.v", "weight_cache.v",
                     "rom_sync.v"]
            compile_cmd = (
                ["iverilog", "-g2012", "-s", "tb", "-o", str(output)]
                + [str(ROOT / "fpga" / f) if f == "generation_controller.v"
                   else str(ROOT / "hdl" / f) for f in files]
                + [str(tb_path)]
            )
            result = subprocess.run(compile_cmd, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            result = subprocess.run(["vvp", str(output)], capture_output=True,
                                    text=True, timeout=timeout, cwd=tmp)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

        line = next(l for l in result.stdout.splitlines() if l.startswith("THROUGHPUT"))
        fields = dict(kv.split("=") for kv in line.split()[1:])
        cycles = int(fields["cycles"])
        tokens = int(fields["tokens"])
        self.assertGreater(cycles, 0)
        self.assertEqual(tokens, gen_tokens)
        return cycles, tokens

    def report(self, label, cycles, tokens, dim, layers):
        cpt = cycles / tokens
        print(f"\n[throughput] {label}: {tokens} tokens, {cycles} cycles, "
              f"{cpt:.0f} cycles/token (d_model={dim}, layers={layers})")
        for name, hz in CLOCK_HZ.items():
            print(f"[throughput]   {name}: {hz / cpt:,.1f} tok/s")
        return cpt

    @unittest.skipUnless(shutil.which("iverilog") and shutil.which("vvp"),
                         "requires Icarus Verilog")
    def test_throughput_small(self):
        rng = random.Random(0xC0DE)
        dim, heads, length, layers, d_ff, vocab = 4, 2, 8, 2, 4, 8
        gen_tokens = 4
        model = GPTControllerTests().build_model(
            rng, dim=dim, heads=heads, length=length, layers=layers, d_ff=d_ff,
            vocab=vocab, shifts=(0, 0, 0, 0))
        cycles, tokens = self.measure(model, [4, 5, 6], dim, heads, length,
                                      layers, d_ff, vocab, gen_tokens)
        cpt = self.report("small", cycles, tokens, dim, layers)
        # Sanity bound: each token is a full forward pass, so it must cost more
        # than one attention weight-load (4*dim*dim cycles) per layer.
        self.assertGreater(cpt, 4 * dim * dim * layers)

    @unittest.skipUnless(shutil.which("iverilog") and shutil.which("vvp"),
                         "requires Icarus Verilog")
    def test_throughput_packed(self):
        # Resident MLP caches + NUM_PES-wide PE array: log cycles/token for the
        # packed path and confirm it is no worse than the scalar baseline.
        rng = random.Random(0xC0DE)
        dim, heads, length, layers, d_ff, vocab = 4, 2, 8, 2, 4, 8
        gen_tokens = 4
        model = GPTControllerTests().build_model(
            rng, dim=dim, heads=heads, length=length, layers=layers, d_ff=d_ff,
            vocab=vocab, shifts=(0, 0, 0, 0))
        base, _ = self.measure(model, [4, 5, 6], dim, heads, length, layers,
                               d_ff, vocab, gen_tokens)
        packed, _ = self.measure(model, [4, 5, 6], dim, heads, length, layers,
                                 d_ff, vocab, gen_tokens, num_pes=4)
        self.report("packed", packed, gen_tokens, dim, layers)
        self.assertLessEqual(packed, base)

    @unittest.skipUnless(shutil.which("iverilog"), "requires Icarus Verilog")
    def test_board_top_elaborates_with_pll(self):
        """fpga_top + pll_150 + board_params.vh elaborate with the PLL enabled.

        Uses the altera_pll blackbox stub (tests/hdl_stubs) so the 150 MHz PLL
        path can be checked without Quartus.
        """
        outdir = tempfile.TemporaryDirectory(prefix="fpgpt-top-")
        output = pathlib.Path(outdir.name) / "top"
        fpga_files = ["fpga_top.v", "generation_controller.v", "uart_rx.v",
                      "uart_tx.v", "pll_150.v"]
        hdl_files = ["transformer_engine.v", "attention.v", "layer_norm.v",
                     "dense_layer.v", "mac_unit.v", "activation.v", "rom_sync.v",
                     "weight_cache.v"]
        cmd = (
            ["iverilog", "-g2012", "-DFPGPT_USE_PLL", "-I", str(ROOT / "fpga"),
             "-s", "fpga_top", "-o", str(output)]
            + [str(ROOT / "fpga" / f) for f in fpga_files]
            + [str(ROOT / "hdl" / f) for f in hdl_files]
            + [str(ROOT / "tests" / "hdl_stubs" / "altera_pll.v")]
        )
        result = subprocess.run(cmd, capture_output=True, text=True)
        outdir.cleanup()
        self.assertEqual(result.returncode, 0, result.stderr)

    @unittest.skipUnless(shutil.which("iverilog") and shutil.which("vvp"),
                         "requires Icarus Verilog")
    @unittest.skipUnless(os.environ.get("FPGPT_THROUGHPUT_DEFAULT") == "1",
                         "set FPGPT_THROUGHPUT_DEFAULT=1 to run")
    def test_throughput_default_config(self):
        # Default Micro-GPT shape; measures the real per-token cost in RTL.
        rng = random.Random(0x64)
        dim, heads, length, layers, d_ff, vocab = 64, 4, 8, 4, 256, 64
        model = GPTControllerTests().build_model(
            rng, dim=dim, heads=heads, length=length, layers=layers, d_ff=d_ff,
            vocab=vocab, shifts=(0, 0, 0, 0))
        cycles, tokens = self.measure(model, [4, 5, 6], dim, heads, length,
                                      layers, d_ff, vocab, gen_tokens=1,
                                      timeout=600)
        self.report("default", cycles, tokens, dim, layers)


if __name__ == "__main__":
    unittest.main()
