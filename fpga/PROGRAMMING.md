# fpGPT board bring-up — DE1-SoC

This is the runbook for turning a compiled model into a programmed DE1-SoC.
It covers ROM/parameter sync, the Quartus flow, timing/area measurement,
programming, and the UART smoke test.

> Quartus Prime has not been run against this design in CI; the flow below is
> the intended one and the numbers it produces are not yet recorded.

> Historical bring-up closed timing at 62.5 MHz with `NUM_PES=4`, but its
> source ROM was not initialized during synthesis. Rebuild and check timing
> and resource usage with the corrected ROM initialization and actual weights.
> The build now rejects missing or incomplete compiler artifacts.

## 0. Prerequisites

- Intel Quartus Prime Lite 20.1 or newer (Cyclone V support).
- A compiled model: `python compile.py --model checkpoints/micro_gpt.pt --out build/`
  from the repository root. This writes `build/weights/engine/weights_unified.hex`
  (the uniform-width image the engine consumes) and `build/rtl/board_params.vh`.
  `build.tcl` copies both into `fpga/`; the checked-in `fpga/board_params.vh` is
  only a placeholder until then.
- An external **3.3 V** USB-to-TTL serial adapter (FTDI/CP2102) or an
  ESP32-S2 USB-UART bridge for the FPGA UART. The onboard USB-UART is wired to
  the HPS, not the FPGA (DE1-SoC User Manual §3.7.3; HPS_UART on `PIN_B25`/
  `PIN_C25`), so it cannot be used to reach `UART_RXD`/`UART_TXD`.
- The onboard JTAG cable. It enumerates as **`DE-SoC`**, not `USB-Blaster`.

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

The generated header and ROM must both exist and the ROM word count must match
`FPGPT_ROM_DEPTH`. Missing or malformed artifacts stop the build; it no longer
substitutes an all-zero image. For validation and synchronization without
Quartus, run `FPGPT_PREPARE_ONLY=1 tclsh fpga/build.tcl` from the repository root.
The build also derives `weights_unified.hex.mif` from that validated hex image.
`rom_sync.v` uses `$readmemh` for simulation and the `ram_init_file` attribute
for synthesis, so Quartus loads the same weights without expanding a large HDL
initialization block. This follows the vendor's
[inferred memory initialization guidance](https://docs.altera.com/r/docs/683283/18.1/quartus-prime-standard-edition-user-guide/ram-initialization-file-for-inferred-memory).

### Prompt and output capacity

Requests are independent. The controller clears the sequence and KV prefix
length after the final output byte is accepted by UART; no reset button is
needed between prompts. Wait for all response bytes before sending the next
prompt. Input received during generation is not queued.

The prompt limit is `MAX_SEQ_LEN - GEN_TOKENS`. The current trained checkpoint
has 64 learned positions, allowing 48 prompt characters and 16 output
characters. The controller retains the first 48 characters of an oversized
prompt, discards the rest through the newline, and lights LEDR[4]. That flag
clears on the first character of the next prompt. The host GUI rejects an
oversized prompt before transmission. Empty lines do not generate output.

Change the response count through the compiler, for example:

```bash
python compile.py --model checkpoints/micro_gpt.pt --out build/ --gen-tokens 32
```

With the existing checkpoint this allows **32 prompt + 32 output** characters;
it does not increase total context. The compiler rejects output counts that
leave no prompt space. The host's expected response count must match the newly
compiled header and programmed bitstream.

To increase both limits, train a new 128-position model in a separate directory:

```bash
python -m model.train --data data/sample.txt --seq_len 128 --vocab_size 64 --output_dir checkpoints/context128
python compile.py --model checkpoints/context128/micro_gpt.pt --out build/ --gen-tokens 32
cd fpga
quartus_sh -t build.tcl
```

This configures **96 prompt + 32 output** characters. Do not just change
`FPGPT_MAX_SEQ_LEN`: positional embeddings and all following ROM offsets depend
on the checkpoint's context size. `--resume` restores the old model geometry;
it does not extend a 64-position checkpoint. The 128/32 controller configuration
is tested using a small synthetic model; fit and timing for a trained larger
model still need verification in Quartus. Increasing context also increases
activation/KV memory and attention work.

The engine retains only the current layer's MLP weights in its packed caches.
Keeping all four layers there alongside the initialized source ROM required
418 M10K blocks on this 397-block device. Reloading the caches between layers
reduces duplicate weight storage; the model arithmetic and KV cache remain
unchanged, but generation includes an additional weight-transfer cost.

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

On Linux the USB-Blaster/DE-SoC needs a udev rule so the device node is
writable (otherwise `jtagconfig` reports "No JTAG hardware available"):

```bash
sudo tee /etc/udev/rules.d/92-usbblaster.rules >/dev/null <<'EOF'
SUBSYSTEMS=="usb", ATTRS{idVendor}=="09fb", MODE="0666"
EOF
sudo udevadm control --reload-rules
sudo udevadm trigger
```

The DE1-SoC's onboard cable enumerates as **`DE-SoC`** (not `USB-Blaster`) and
its JTAG chain has two devices: the **HPS at index 1** and the **FPGA at index
2**, so the SOF must be targeted at `@2`. Confirm the cable name and chain with:

```bash
jtagconfig
```

```bash
export PATH="/home/jcmb/altera_lite/25.1std/quartus/bin:$PATH"
quartus_pgm -c "DE-SoC" -m jtag -o "p;output_files/fpGPT.sof@2"
```

or in the GUI: **Tools → Programmer → Add File → Start**.

If `jtagconfig` prints "No JTAG hardware available", the udev rule above is not
in effect (re-plug the board, or re-run `udevadm trigger`). If programming fails
with a device-index error, re-run `jtagconfig` — the HPS/FPGA ordering (`@1`/
`@2`) is what the DE-SoC chain reports.

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

| Signal | FPGA pin | GPIO_0 | Physical header |
| --- | --- | --- | --- |
| `UART_RXD` | `PIN_AC18` | `GPIO_0[0]` | pin 1 (adapter TX) |
| `UART_TXD` | `PIN_Y17` | `GPIO_0[1]` | pin 2 (adapter RX) |
| GND | — | GND | pin 12 or pin 30 |

`GPIO_0` is a 2×20-pin (40-pin) expansion header on the right edge of the
board. The `GPIO_0[n]` name is a 0-based signal index, not the physical pin
number: physical pin 1 is `GPIO_0[0]`, pin 2 is `GPIO_0[1]`, and so on. Ground
on the header is available at physical pins 12 and 30.

Do **not** connect the adapter's VCC if the board is already powered; share
ground only. Settings: **115200 8N1, no flow control**.

### ESP32-S2 USB-UART bridge (alternative)

A WEMOS LOLIN S2 Mini running the bridge sketch can act as the USB-UART adapter
instead of an FTDI/CP2102. In the Arduino IDE select Board **"ESP32S2 Dev
Module"** with **USB CDC On Boot: Enabled**; baud is **115200**.

| ESP32-S2 | Direction | DE1-SoC |
| --- | --- | --- |
| `GPIO17` (TX) | → | `UART_RXD` = `GPIO_0[0]` = `PIN_AC18` (physical pin 1) |
| `GPIO18` (RX) | ← | `UART_TXD` = `GPIO_0[1]` = `PIN_Y17` (physical pin 2) |
| `GND` | — | `GND` (physical pin 12 or 30) |

On the S2 Mini's right-side header, `GPIO17` is the inner row (the row directly
above GND) and `GPIO18` is the outer row (the row containing `GPIO21`). Both
sides are 3.3 V — never connect 5 V. See `esp32s2_webserver/README.md` for the
WiFi bridge firmware.

## 6. Smoke test

1. Open a serial terminal (e.g. `screen /dev/tty.usbserial-* 115200`).
2. Press `KEY[0]` to release reset (LEDR[0] off, LEDR[1] = PLL locked).
3. Type a short prompt and press Enter (CR/LF). The `generation_controller`
   starts decoding when it sees CR or LF.
4. The board streams exactly `GEN_TOKENS` generated characters back. Send
   several prompts without pressing reset and check that every response has
   the configured length. The byte count, rather than the generated text, is
   the initial transport check.
5. LEDs: LEDR[2] pulses per generated character, LEDR[3] is high while the
   engine is busy, LEDR[4] indicates prompt truncation, and LEDR[9:5] show the
   low five bits of the last received byte.

Because the current model is a small, lightly trained character model, output
quality is expected to be poor; the test is that the pipeline runs end to end.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| `No JTAG hardware available` | Missing/inactive udev rule (09fb); re-plug or `udevadm trigger` |
| Unknown cable name in `quartus_pgm` | Use `-c "DE-SoC"`, not `-c "USB-Blaster"` |
| Programming fails on device index | DE-SoC chain has HPS `@1`, FPGA `@2`; target the SOF with `@2` |
| `jtagconfig` shows two devices | Expected: HPS at index 1, FPGA at index 2 |
| No UART output | PLL locked (LEDR[1]), TX/RX not swapped, common ground, 3.3 V logic |
| Output is `.` only | Check the actual ROM/checkpoint and synthesis report; missing weights now stop the build |
| Only one character after several prompts | Rebuild/program the controller fix; old bitstreams retain a full context until KEY[0] resets it |
| LEDR[4] lit | Prompt exceeded `MAX_SEQ_LEN - GEN_TOKENS`; only its prefix was retained |
| Garbage characters | Baud mismatch or wrong `CLK_FREQ` (must match the PLL output, 62.5 MHz) |
| Timing not met | Lower the PLL frequency in `pll_150.v`, re-run `timing.tcl` |
| Fit fails on M10K | ROM depth exceeds 556 KB; reduce the model or use INT4 |
