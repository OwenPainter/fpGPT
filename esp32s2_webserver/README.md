# fpGPT ESP32-S2 webserver

A WiFi webserver for an **ESP32-S2** dev board, flashed with the **Arduino IDE**.
It serves a chat page and bridges prompts to the fpGPT design running on the
DE1-SoC FPGA over UART:

```
Browser --HTTP--> ESP32-S2 --UART 115200 8N1--> DE1-SoC (fpga_top.v)
```

It replaces the PC-based `host_gui/` bridge with a standalone WiFi device, so
you can chat with the model from a phone or laptop without a host computer.

## Files

| File | Purpose |
| --- | --- |
| `esp32s2_webserver.ino` | Sketch: WiFi, HTTP server, streaming bridge |
| `config.h` | WiFi credentials, UART pins/baud, link mode |
| `fpga_link.h` | UART framing (raw + framed protocol codec) |
| `web_page.h` | Embedded HTML/CSS/JS chat UI (served from flash) |

## 1. Arduino IDE setup

1. Install **Arduino IDE 2.x**.
2. **File → Preferences → Additional Boards Manager URLs**, add:
   `https://espressif.github.io/arduino-esp32/package_esp32_index.json`
3. **Tools → Board → Boards Manager**, search `esp32`, install **esp32 by
   Espressif Systems**.
4. Select **Tools → Board → ESP32S2 Dev Module**.
5. Recommended Tools settings:
   - **USB CDC On Boot: Enabled** (so `Serial` is the USB serial monitor)
   - **Upload Speed: 921600**
   - **Flash Size / PSRAM**: match your board
6. Open `esp32s2_webserver/esp32s2_webserver.ino`.

> Some ESP32-S2 boards need **BOOT** held while tapping **RESET** to enter the
> ROM bootloader for the first upload.

## 2. Configure

Edit `config.h`:

```c
#define WIFI_SSID     "your-ssid"
#define WIFI_PASSWORD "your-password"

#define FPGA_BAUD     115200
#define FPGA_RX_PIN   18   // ESP32 RX  <- FPGA TX
#define FPGA_TX_PIN   17   // ESP32 TX  -> FPGA RX

#define FPGA_FRAMED   0    // 0 = raw (current firmware), 1 = framed protocol
#define FPGA_MOCK     0    // 1 = reply locally, no FPGA needed
```

Set `FPGA_MOCK 1` to try the page without any hardware.

## 3. Wire to the DE1-SoC

The FPGA UART is on `GPIO_0[0:1]` (see `fpga/PROGRAMMING.md`), 3.3 V logic:

| ESP32-S2 | DE1-SoC | Notes |
| --- | --- | --- |
| `FPGA_RX_PIN` (GPIO18) | `UART_TXD` / `GPIO_0[1]` / `PIN_Y17` | ESP32 receives |
| `FPGA_TX_PIN` (GPIO17) | `UART_RXD` / `GPIO_0[0]` / `PIN_AC18` | ESP32 transmits |
| GND | GND | **share ground** |

Both sides are 3.3 V — do not connect 5 V. Power the ESP32 from its own USB.

## 4. Flash and run

1. Upload the sketch (**Upload** button).
2. Open **Tools → Serial Monitor** at **115200**; note the printed IP, e.g.
   `[wifi] connected, IP 192.168.1.42`.
3. Browse to `http://<ip>/` on any device on the same network.
4. Type a prompt and press **Enter**. The reply streams in as the FPGA generates.

## 5. HTTP API

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/` | Chat page |
| `GET` | `/api/status` | JSON: `wifi`, `ip`, `rssi`, `baud`, `framed`, `mock`, `decoder_errors` |
| `POST` | `/api/generate` | Body is the prompt text; response is streamed `text/plain` |
| `POST` | `/api/reset` | Reset/flush the FPGA link |

Example:

```bash
curl -N -X POST --data 'To be' http://<ip>/api/generate
```

## 6. Link modes

**Raw (default, `FPGA_FRAMED 0`)** — matches the current
`hdl/generation_controller.v`: the sketch sends the prompt bytes followed by
`\n`, the firmware decodes and streams raw characters back. Because the raw
firmware has no end marker, the bridge ends the reply after
`FPGA_IDLE_MS` (1500 ms) of silence or `FPGA_MAX_MS` overall.

**Framed (`FPGA_FRAMED 1`)** — the framed protocol defined in
`host_gui/fgpt_gui/protocol.py` and `docs/frontend_gui_plan.md`:

```
STX(0x02) | LEN(1) | CMD(1) | PAYLOAD[LEN] | CKSUM(1) | ETX(0x03)
CKSUM = XOR of LEN, CMD and PAYLOAD
```

Host→FPGA `PROMPT`/`GENERATE`/`RESET`/`PING`/`INFO`; FPGA→host `TOKEN`/`DONE`/
`ERROR`/`READY`/`INFO_RESP`. Use this once the FPGA firmware implements it.

## 7. Troubleshooting

| Symptom | Check |
| --- | --- |
| Page unreachable | Same 2.4 GHz network (ESP32-S2 is 2.4 GHz only); check the IP in the serial monitor |
| `decoder_errors` climbing | Baud mismatch or wrong pin mapping (framed mode) |
| Reply is `.` only | FPGA ROM uninitialized / all-zero weights (`fpga/PROGRAMMING.md`) |
| Reply truncated | Increase `FPGA_IDLE_MS`; the raw firmware has no terminator |
| Upload fails | Hold **BOOT**, tap **RESET**, retry; enable **USB CDC On Boot** |

## Limitations

- HTTP handling is single-threaded: `/api/generate` blocks while the FPGA runs,
  so concurrent requests wait.
- No authentication or TLS — use only on a trusted LAN.
- One request at a time; a reply must finish before the next prompt.
