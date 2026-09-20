"""Board UART integration and build artifact validation."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class BoardFlowTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which('iverilog') and shutil.which('vvp'), 'requires Icarus')
    def test_repeated_uart_requests(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'sim'
            sources = [ROOT/'tests/tb_fpga_top.v'] + [ROOT/'fpga'/name for name in
                       ('fpga_top.v', 'generation_controller.v', 'uart_rx.v', 'uart_tx.v')]
            sources += list((ROOT/'hdl').glob('*.v'))
            subprocess.run(['iverilog', '-g2012', '-s', 'tb_fpga_top', '-o', str(output)]
                           + list(map(str, sources)), check=True, capture_output=True)
            result = subprocess.run(['vvp', str(output)], cwd=tmp, capture_output=True,
                                    text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn('four prompts, two UART bytes each', result.stdout)

    @unittest.skipUnless(shutil.which('tclsh'), 'requires Tcl')
    def test_build_rejects_missing_or_partial_rom(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'fpga').mkdir()
            shutil.copy(ROOT/'fpga/build.tcl', root/'fpga/build.tcl')
            env = dict(os.environ, FPGPT_PREPARE_ONLY='1')

            def run():
                return subprocess.run(['tclsh', str(root/'fpga/build.tcl')],
                                      env=env, capture_output=True, text=True)

            self.assertNotEqual(run().returncode, 0)
            (root/'build/rtl').mkdir(parents=True)
            (root/'build/weights/engine').mkdir(parents=True)
            (root/'build/rtl/board_params.vh').write_text(
                '`define FPGPT_ROM_DEPTH 4\n`define FPGPT_DATA_WIDTH 8\n'
                '`define FPGPT_MAX_SEQ_LEN 64\n`define FPGPT_GEN_TOKENS 16\n')
            rom = root/'build/weights/engine/weights_unified.hex'
            rom.write_text('00\n')
            self.assertNotEqual(run().returncode, 0)
            self.assertFalse((root/'fpga/weights_unified.hex').exists())
            rom.write_text('00\n12\n34\nGG\n')
            self.assertNotEqual(run().returncode, 0)
            rom.write_text('00\n12\n34\nff\n')
            result = run()
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual((root/'fpga/weights_unified.hex').read_text(), rom.read_text())
            mif = (root/'fpga/weights_unified.hex.mif').read_text()
            self.assertIn('WIDTH=8;\nDEPTH=4;', mif)
            self.assertIn('0 : 00;\n1 : 12;\n2 : 34;\n3 : ff;', mif)
