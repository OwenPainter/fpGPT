# Bring-up log — DE1-SoC

Verified on hardware (Terasic DE1-SoC, Cyclone V `5CSEMA5F31C6`). Facts only;
see `fpga/PROGRAMMING.md` for the runbook.

## Board and cabling

- Onboard USB-UART is wired to the **HPS**, not the FPGA (User Manual §3.7.3;
  `HPS_UART` on `PIN_B25`/`PIN_C25`). An external 3.3 V UART is required to
  reach the FPGA design.
- FPGA UART: **115200 8N1**, 3.3 V.
  - `UART_RXD` = `GPIO_0[0]` = `PIN_AC18` — physical header pin 1.
  - `UART_TXD` = `GPIO_0[1]` = `PIN_Y17` — physical header pin 2.
  - GND at physical header pin 12 or 30.
- `GPIO_0` is a 2×20-pin (40-pin) header on the right edge of the board; the
  `GPIO_0[n]` index is 0-based and is not the physical pin number (pin 1 =
  `GPIO_0[0]`, pin 2 = `GPIO_0[1]`).

## JTAG programming

- The onboard cable enumerates as **`DE-SoC`** (not `USB-Blaster`) with a
  two-device chain: **HPS at index 1**, **FPGA at index 2**.
- Working command:
  ```bash
  quartus_pgm -c "DE-SoC" -m jtag -o "p;output_files/fpGPT.sof@2"
  ```
- Linux udev rule required for a writable device node:
  ```bash
  sudo tee /etc/udev/rules.d/92-usbblaster.rules >/dev/null <<'EOF'
  SUBSYSTEMS=="usb", ATTRS{idVendor}=="09fb", MODE="0666"
  EOF
  sudo udevadm control --reload-rules && sudo udevadm trigger
  ```

## ESP32-S2 bridge (alternative to FTDI/CP2102)

- WEMOS LOLIN S2 Mini, Arduino IDE Board **"ESP32S2 Dev Module"**,
  **USB CDC On Boot: Enabled**, 115200.
- `GPIO17` (TX) → FPGA `GPIO_0[0]`; `GPIO18` (RX) ← FPGA `GPIO_0[1]`; GND
  shared. On the S2 Mini right-side header, `GPIO17` is the inner row (directly
  above GND) and `GPIO18` is the outer row (the row containing `GPIO21`).

## Clock and observed bring-up

- System clock is **62.5 MHz** (the PLL module is still named `pll_150.v`).
- With an uninitialized/all-zero ROM, the verified bring-up output over UART is
  a stream of `.` characters — this exercises the clock/reset/UART/FSM path.