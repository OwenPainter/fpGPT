"""End-to-end test for the board-level generation controller.

Drives fpga/generation_controller.v the way UART would (token_in/token_valid),
with a tiny model ROM, and checks that the emitted byte is the character for
the argmax of the integer reference's final-position logits. This exercises the
whole wiring: prompt buffering, character<->token mapping, transformer_engine,
the weight ROM, logit capture, argmax and the output mapping.

Run: python3 -m unittest discover -s tests -v
"""
import pathlib
import random
import shutil
import subprocess
import tempfile
import unittest

from tests.test_transformer_engine import GPTControllerTests

ROOT = pathlib.Path(__file__).resolve().parents[1]


class GenerationControllerTests(unittest.TestCase):
    def simulate(self, model, tokens, dim, heads, length, layers, d_ff, vocab,
                 shifts=(0, 0, 0, 0), mult=64, score_shift=8, ln_shift=7,
                 gen_tokens=1, prompts=None, stall_output=False):
        helper = GPTControllerTests()

        def argmax_id(seq):
            logits = helper.reference(model, seq, dim, heads, d_ff, shifts, mult,
                                      score_shift, ln_shift)
            return max(range(len(logits)), key=lambda i: logits[i])

        def to_byte(tok_id):
            return (tok_id - 3 + 32) if tok_id >= 3 else ord('.')

        # Mirror the controller: run the engine, emit argmax, append and repeat.
        prompts = prompts or [tokens]
        expected_bytes = []
        for prompt in prompts:
            seq = list(prompt[:length-gen_tokens])
            for _ in range(gen_tokens):
                best_id = argmax_id(seq)
                expected_bytes.append(to_byte(best_id))
                seq.append(best_id)
        total_tokens = len(expected_bytes)

        image = helper.rom_image(model, dim, length, vocab, d_ff)
        w_addr_width = max(1, (len(image) - 1).bit_length())

        # Prompt token IDs must be >= 3 so the byte mapping round-trips.
        prompt_send = []
        for request, prompt in enumerate(prompts):
            assert all(t >= 3 for t in prompt), "prompt tokens must map to characters"
            for t in prompt:
                prompt_send.append(f"@(negedge clk); token_in={t-3+32}; token_valid=1;\n"
                                   "@(negedge clk); token_valid=0;")
            overflow = int(len(prompt) > length-gen_tokens)
            prompt_send.append(f"""
    @(negedge clk); token_in=13; token_valid=1;
    @(negedge clk); token_in=10;
    @(negedge clk); token_valid=0;
    wait(got == {(request+1)*gen_tokens});
    @(negedge clk);
    if (busy || dut.seq_len != 0 || dut.cache_len != 0)
        $fatal(1,"request did not clear its context");
    if (dut.prompt_overflow !== 1'b{overflow}) $fatal(1,"overflow flag mismatch");
    // A blank line after completion must not regenerate the old context.
    token_in=10; token_valid=1;
    @(negedge clk); token_valid=0;
    repeat(20) @(negedge clk);
    if (busy || got != {(request+1)*gen_tokens}) $fatal(1,"blank line triggered generation");
""")
        prompt_send = "\n".join(prompt_send)
        hex_lines = "\n".join(f"{v & 0xFF:02X}" for v in image)
        expected_checks = "\n".join(
            f"expected_bytes[{i}] = {v};" for i, v in enumerate(expected_bytes)
        )
        compare = "\n".join(
            f"    if (captured[{i}] !== expected_bytes[{i}]) "
            f"$fatal(1, \"token %0d got %0d expected %0d\", {i}, captured[{i}], expected_bytes[{i}]);"
            for i in range(total_tokens)
        )

        tb = f'''
module tb;
reg clk=0; always #5 clk=~clk;
reg rst_n=0;
reg [7:0] token_in=0;
reg token_valid=0;
wire [7:0] token_out;
wire token_out_valid, busy;
integer got=0;
reg [7:0] captured [0:{total_tokens-1}];
integer expected_bytes [0:{total_tokens-1}];
integer cycle=0;
wire token_ready = {"(cycle % 8192) >= 6000" if stall_output else "1'b1"};
always @(negedge clk) cycle=cycle+1;
reg stalled=0;
reg [7:0] held;

generation_controller #(
    .DATA_WIDTH(8), .ACC_WIDTH(32), .D_MODEL({dim}), .NUM_HEADS({heads}),
    .MAX_SEQ_LEN({length}), .NUM_LAYERS({layers}), .D_FF({d_ff}),
    .VOCAB_SIZE({vocab}),
    .Q_SHIFT({shifts[0]}), .K_SHIFT({shifts[1]}), .V_SHIFT({shifts[2]}),
    .OUT_SHIFT({shifts[3]}), .SCORE_MULT({mult}), .SCORE_SHIFT({score_shift}),
    .LN_SHIFT({ln_shift}), .FC1_SHIFT(0), .FC2_SHIFT(0), .LM_SHIFT(0),
    .W_ADDR_WIDTH({w_addr_width}), .ROM_DEPTH({len(image)}),
    .ROM_MEM_FILE("weights.hex"), .GEN_TOKENS({gen_tokens}), .TOP_K(1)
) dut (
    .clk(clk), .rst_n(rst_n), .token_in(token_in), .token_valid(token_valid),
    .token_out(token_out), .token_out_valid(token_out_valid), .busy(busy),
    .token_out_ready(token_ready), .prompt_overflow()
);

always @(posedge clk) begin
    if (stalled && (!token_out_valid || token_out !== held))
        $fatal(1,"output changed under backpressure");
    stalled <= token_out_valid && !token_ready;
    held <= token_out;
    if (token_out_valid && token_ready) begin
        if (got >= {total_tokens}) $fatal(1,"extra output byte");
        captured[got] <= token_out;
        got <= got + 1;
    end
end

initial begin
    {expected_checks}
    repeat(2) @(negedge clk); rst_n=1;
    {prompt_send}
    @(posedge clk);
{compare}
    $display("PASS");
    $finish;
end
initial begin #20000000; $fatal(1,"timeout"); end
endmodule
'''
        with tempfile.TemporaryDirectory(prefix="fpgpt-gen-") as tmp:
            tmp = pathlib.Path(tmp)
            (tmp / "weights.hex").write_text(hex_lines + "\n")
            tb_path = tmp / "tb.v"
            tb_path.write_text(tb)
            output = tmp / "sim"
            files = ["generation_controller.v", "transformer_engine.v", "attention.v",
                     "layer_norm.v", "dense_layer.v", "mac_unit.v", "activation.v", "weight_cache.v",
                     "rom_sync.v"]
            result = subprocess.run(
                ["iverilog", "-g2012", "-s", "tb", "-o", str(output)]
                + [str(ROOT / "fpga" / f) if f == "generation_controller.v"
                   else str(ROOT / "hdl" / f) for f in files] + [str(tb_path)],
                capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            result = subprocess.run(["vvp", str(output)], capture_output=True, text=True,
                                    timeout=120, cwd=tmp)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("PASS", result.stdout)

    @unittest.skipUnless(shutil.which("iverilog") and shutil.which("vvp"),
                         "requires Icarus Verilog")
    def test_single_generated_token(self):
        rng = random.Random(0xA5)
        tokens = [4, 5, 6]
        model = GPTControllerTests().build_model(
            rng, dim=4, heads=2, length=4, layers=2, d_ff=4, vocab=8,
            shifts=(0, 0, 0, 0))
        self.simulate(model, tokens, dim=4, heads=2, length=4, layers=2, d_ff=4,
                      vocab=8, gen_tokens=1)

    @unittest.skipUnless(shutil.which("iverilog") and shutil.which("vvp"),
                         "requires Icarus Verilog")
    def test_autoregressive_feedback(self):
        # Context capacity 6 with 3 prompt tokens leaves room to append 3.
        rng = random.Random(0x3C)
        tokens = [4, 5, 6]
        model = GPTControllerTests().build_model(
            rng, dim=4, heads=2, length=6, layers=1, d_ff=4, vocab=8,
            shifts=(0, 0, 0, 0))
        self.simulate(model, tokens, dim=4, heads=2, length=6, layers=1, d_ff=4,
                      vocab=8, gen_tokens=3)

    @unittest.skipUnless(shutil.which("iverilog") and shutil.which("vvp"), "requires Icarus Verilog")
    def test_repeated_prompts_overflow_and_backpressure(self):
        model = GPTControllerTests().build_model(
            random.Random(317), dim=4, heads=2, length=8, layers=2,
            d_ff=4, vocab=8, shifts=(0, 0, 0, 0))
        self.simulate(model, [4], dim=4, heads=2, length=8, layers=2,
                      d_ff=4, vocab=8, gen_tokens=4, stall_output=True,
                      prompts=[[4, 5, 6, 7], [7]*12, [6], [3, 4, 5, 6]])

    @unittest.skipUnless(shutil.which("iverilog") and shutil.which("vvp"), "requires Icarus Verilog")
    def test_128_context_96_prompt_32_output(self):
        # Synthetic 128-position model: verifies counters and reserved space;
        # it does not claim the checked-in 64-position checkpoint is extended.
        model = GPTControllerTests().build_model(
            random.Random(128), dim=4, heads=2, length=128, layers=1,
            d_ff=4, vocab=8, shifts=(0, 0, 0, 0))
        self.simulate(model, [4]*96, dim=4, heads=2, length=128, layers=1,
                      d_ff=4, vocab=8, gen_tokens=32)


if __name__ == "__main__":
    unittest.main()
