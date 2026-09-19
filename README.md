# fpGPT — ML Weights to Silicon Compiler

**Transform trained neural network weights into synthesizable Verilog RTL that runs directly on FPGA hardware — no CPU, no OS, no software runtime.**

Inspired by [Taalas/ChatJimmy](https://chatjimmy.ai), which physically bakes LLM weights into ASIC silicon for 15,000+ tokens/sec inference. fpGPT achieves the same "model-on-silicon" architecture using reconfigurable FPGA fabric.

## Target Hardware
- **Board:** Terasic DE1-SoC
- **FPGA:** Intel/Altera Cyclone V 5CSEMA5F31C6
- **On-Chip Memory:** 556 KB M10K SRAM (weights burned directly into block RAM)
- **Compute:** 87 DSP blocks (18×18 signed multipliers)
- **Clock:** 50 MHz (PLL configurable to ~150 MHz)

## Architecture

```
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  PyTorch Model   │ ──► │  fpGPT Compiler  │ ──► │  Verilog RTL +   │
│  (micro_gpt.pt)  │     │  (quantize+emit) │     │  .mif weight ROM │
└──────────────────┘     └──────────────────┘     └──────────────────┘
                                                          │
                                                          ▼
                                                  ┌──────────────────┐
                                                  │  Quartus Synth   │
                                                  │  → FPGA Bitstream│
                                                  │  → Hardware AI!  │
                                                  └──────────────────┘
```

## Quick Start

### 1. Train a Micro-GPT Model
```bash
# Train on built-in Shakespeare text (quick test)
python -m model.train --epochs 50

# Train on your own text corpus
python -m model.train --data data/input.txt --epochs 100
```

### 2. Compile to Verilog
```bash
python compile.py --model checkpoints/micro_gpt.pt --out build/
```

This generates:
- `build/weights/*.mif` — M10K memory initialization files
- `build/weights/*.hex` — Verilog `$readmemh` files
- `build/rtl/gpt_params.vh` — Auto-generated parameters header
- `build/rtl/weight_rom.v` — Unified weight ROM module

### 3. Synthesize with Quartus
Open Quartus, create a project targeting the Cyclone V 5CSEMA5F31C6, add the RTL files from `hdl/` and `build/rtl/`, and compile.

## Project Structure
```
fpGPT/
├── compiler/           # Python compilation pipeline
│   ├── ir.py           #   Intermediate representation
│   ├── quantizer.py    #   FP32 → INT8 weight quantization
│   └── mif_writer.py   #   .mif / .hex file generator
├── model/              # PyTorch model definition & training
│   ├── micro_gpt.py    #   Character-level GPT architecture
│   └── train.py        #   Training script
├── hdl/                # Synthesizable Verilog RTL
│   ├── mac_unit.v      #   DSP Multiply-Accumulate unit
│   ├── rom_sync.v      #   M10K block RAM ROM
│   ├── activation.v    #   ReLU / GELU hardware functions
│   └── dense_layer.v   #   Matrix-vector multiply engine
├── compile.py          # CLI compiler entry point
└── requirements.txt
```

## Default Model Configuration
| Parameter | Value | Notes |
|-----------|-------|-------|
| Vocab size | 64 | Printable ASCII subset |
| Context length | 64 | Tokens per inference window |
| d_model | 64 | Embedding dimension |
| Attention heads | 4 | Head dim = 16 |
| Transformer layers | 4 | Pre-norm decoder blocks |
| MLP hidden dim | 256 | 4× expansion ratio |
| **Total params** | **~210K** | **~210 KB at INT8** |
| **M10K usage** | **~38%** | Well within 556 KB budget |

## License
MIT
