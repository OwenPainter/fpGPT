# fpGPT board bring-up — DE1-SoC

This is the runbook for turning a compiled model into a programmed DE1-SoC.
It covers ROM/parameter sync, the Quartus flow, timing/area measurement,
programming, and the UART smoke test.

> Quartus Prime has not been run against this design in CI; the flow below is
> the intended one and the numbers it produces are not yet recorded.

> **Current status (bring-up):** the design compiles and closes timing at
> 62.5 MHz with `NUM_PES=4`. If no compiled model is present, `build.tcl` writes
> an all-zero ROM, so the smoke test only proves the clock/reset/UART/FSM path —
> the board streams `.` characters, not real text.

## 0. Prerequisites

- Intel Quartus Prime Lite 20.1 or newer (Cyclone V support).
- A compiled model: `python compile.py --model checkpoints/micro_gpt.pt --out build/`
  from the repository root. This writes `build/weights/engine/weights_unified.hex`
  (the uniform-width image the engine consumes) and `build/rtl/board_params.vh`.
  `build.tcl` copies both into `fpga/`; the checked-in `fpga/board_params.vh` is
  only a placeholder until then.
- A 3.3 V USB-to-TTL serial adapter (FTDI/CP2102) for the UART.
- USB-Blaster for JTAG programming.

## 1. Sync generated files and build

`build.tcl` copies the compiler outputs next to the project so `$readmemh` and
`\`include` resolve, then compiles:

```bash
cd fpga
quartus_sh -t build.tcl
```

Outputs:

- `output_files/fpGPT.sof` — SRAM object file for JTAG programming.
- `output_files/fpGPT.fit.rpt` — resource usage (ALMs, DSPs, M10K).
- `output_files/fpGPT.sta.rpt` — timing analysis.

If `board_params.vh` or `weights_unified.hex` are missing, the build still
runs but the ROM is uninitialized (all logits zero, output is `.`).

## 2. Timing and area closure

```bash
cd fpga
quartus_sta -t timing.tcl          # prints Fmax + worst setup paths
grep -A20 "Fitter Summary" output_files/fpGPT.fit.rpt
```

Record the following in the PR/README:

- Fmax for `clk_sys` (current target 62.5 MHz / 16.0 ns).
- ALM count, DSP block count, M10K block count.
- Whether setup and hold are met.

The design originally targeted 150 MHz but the combinational dividers in
`layer_norm.v`/`attention.v` did not close. They were replaced with multi-cycle
restoring units and the PLL was lowered to 62.5 MHz (`pll_150.v`), which closes
with margin (slow-85C setup slack ~+2.8 ns, Fmax ~75 MHz at `NUM_PES=4`).

## 3. Program the FPGA (volatile, JTAG)

```bash
quartus_pgm -c "USB-Blaster" -m jtag -o "p;output_files/fpGPT.sof"
```

or in the GUI: **Tools → Programmer → Add File → Start**.

The `.sof` configures SRAM only; power-cycling reloads the previous image.

## 4. Persistent configuration (optional)

To boot from the on-board configuration flash, convert the `.sof` to a `.pof`
(or `.jic`) for the board's EPCS/EPCQ device:

1. **File → Convert Programming Files**.
2. Programming file type: `Programmer Object File (.pof)`.
3. Configuration device: select the DE1-SoC's serial configuration device.
4. Input file: `output_files/fpGPT.sof`.
5. Generate, then program the `.pof` with the Programmer in Active Serial mode.

Command-line equivalent (adjust the device part to match the board):

```bash
quartus_cpf -c -d <config-device> output_files/fpGPT.sof output_files/fpGPT.pof
```

## 5. UART wiring

The DE1-SoC has no direct FPGA-to-USB UART: the on-board USB-UART is wired to
the HPS. Use an external 3.3 V USB-TTL adapter on the **GPIO_0** header:

| Signal | FPGA pin | GPIO_0 pin | Adapter |
| --- | --- | --- | --- |
| `UART_RXD` | `PIN_AC18` | GPIO_0[0] | adapter TX |
| `UART_TXD` | `PIN_Y17` | GPIO_0[1] | adapter RX |
| GND | — | GND | adapter GND |

Do **not** connect the adapter's VCC if the board is already powered; share
ground only. Settings: **115200 8N1, no flow control**.

## 6. Smoke test

1. Open a serial terminal (e.g. `screen /dev/tty.usbserial-* 115200`).
2. Press `KEY[0]` to release reset (LEDR[0] off, LEDR[1] = PLL locked).
3. Type a short prompt and press Enter (CR/LF). The `generation_controller`
   starts decoding when it sees CR or LF.
4. The board streams `GEN_TOKENS` generated characters back.
5. LEDs: LEDR[2] pulses per generated character, LEDR[3] is high while the
   engine is busy, LEDR[9:4] show the last received byte.

Because the current model is a small, lightly trained character model, output
quality is expected to be poor; the test is that the pipeline runs end to end.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| No UART output | PLL locked (LEDR[1]), TX/RX not swapped, common ground, 3.3 V logic |
| Output is `.` only | ROM not initialized (`weights_unified.hex` missing) or all-zero weights |
| Garbage characters | Baud mismatch or wrong `CLK_FREQ` (must match the PLL output, 62.5 MHz) |
| Timing not met | Lower the PLL frequency in `pll_150.v`, re-run `timing.tcl` |
| Fit fails on M10K | ROM depth exceeds 556 KB; reduce the model or use INT4 |
