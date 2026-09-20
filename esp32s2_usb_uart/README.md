# ESP32-S2 USB ↔ UART bridge

Turns an **ESP32-S2** dev board into a plain USB-to-UART adapter: bytes from the
native USB port go out the hardware UART, and UART bytes come back to USB. No
WiFi, no web server, no framing, no dependencies on the rest of this repo.

```
Computer  --USB CDC-->  ESP32-S2  --UART TX/RX-->  target device
```

## Files

| File | Purpose |
| --- | --- |
| `esp32s2_usb_uart.ino` | Sketch: transparent bidirectional byte pump |
| `config.h` | UART pins/baud, USB baud, optional activity LED |

## 1. Arduino IDE setup

1. **Tools → Board → ESP32S2 Dev Module**.
2. **Tools → USB CDC On Boot → Enabled** — required; makes `Serial` the USB port.
3. Upload Speed **921600**. Flash/PSRAM: match your board.
4. Open `esp32s2_usb_uart/esp32s2_usb_uart.ino` and upload.

> Some boards need **BOOT** held while tapping **RESET** for the first upload.

## 2. Wire the target

| ESP32-S2 | Target | Notes |
| --- | --- | --- |
| `UART_RX_PIN` (GPIO18) | target TX | ESP32 receives |
| `UART_TX_PIN` (GPIO17) | target RX | ESP32 transmits |
| GND | GND | **share ground** |

3.3 V logic only — do not feed 5 V. Power the ESP32 over USB.

Edit pins/baud in `config.h`:

```c
#define UART_BAUD     115200
#define UART_RX_PIN   18
#define UART_TX_PIN   17
```

The ESP32-S2 USB port is only for data + power; use a separate USB-UART or the
ROM bootloader for flashing if your board has a single USB connector.

## 3. Use it

Open the host serial monitor or any terminal at 115200 and talk to the target
directly. The bridge is binary-safe and adds nothing to the stream, so it works
for text or raw bytes (e.g. `screen`, `minicom`, `pyserial`, `curl` over a
pseudo-tty).

```powershell
# example: PowerShell / .NET SerialPort, or just use the Arduino Serial Monitor
```

Tip: while bridging, don't print debug text to `Serial` — it would be injected
into the data stream. Keep `FOLLOW_USB_BAUD 0` unless you want the terminal's
chosen baud to drive the UART.

## 4. Options (`config.h`)

| Define | Meaning |
| --- | --- |
| `UART_BAUD` | UART speed (300 – 2000000) |
| `UART_RX_PIN` / `UART_TX_PIN` | Any free GPIO |
| `UART_RX_BUFFER` / `UART_TX_BUFFER` | Driver ring buffers (bytes) |
| `FOLLOW_USB_BAUD` | `1` = adopt host monitor baud and re-clock the UART |
| `LED_PIN` | Optional activity LED GPIO, `-1` disables |
| `BRIDGE_CHUNK` | Bytes copied per loop (throughput/latency tradeoff) |

## Troubleshooting

| Symptom | Check |
| --- | --- |
| Nothing arrives | **USB CDC On Boot** enabled; correct COM port; shared GND |
| Garbage / framing errors | Baud mismatch between host side and target |
| Only one direction works | TX/RX swapped: ESP RX → target TX, ESP TX → target RX |
| Upload fails | Hold **BOOT**, tap **RESET**, retry |
| Data looks truncated | Increase `UART_RX_BUFFER` / `UART_TX_BUFFER` at high baud |