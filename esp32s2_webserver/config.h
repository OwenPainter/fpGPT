// fpGPT ESP32-S2 webserver — configuration.
//
// Edit this file before flashing. Everything else in the sketch has sane
// defaults; only WIFI_SSID / WIFI_PASSWORD must be set.

#pragma once

// ── WiFi ──────────────────────────────────────────────────────────────
#define WIFI_SSID       "your-ssid"
#define WIFI_PASSWORD   "your-password"
#define WIFI_HOSTNAME   "fpgpt-esp32s2"
// How long to wait for a connection before starting the (unconnected) server.
#define WIFI_TIMEOUT_MS 20000

// ── UART to the DE1-SoC FPGA ──────────────────────────────────────────
// 3.3 V logic. ESP32-S2 can route UART1 to almost any GPIO.
//   ESP32 RX (FPGA_RX_PIN) <- FPGA UART_TXD (GPIO_0[1] / PIN_Y17)
//   ESP32 TX (FPGA_TX_PIN) -> FPGA UART_RXD (GPIO_0[0] / PIN_AC18)
//   GND <-> GND
#define FPGA_BAUD       115200
#define FPGA_RX_PIN     18
#define FPGA_TX_PIN     17

// Link mode:
//   0 = raw: prompt bytes + CR/LF, stream raw reply bytes. This matches the
//       current hdl/generation_controller.v firmware.
//   1 = framed: STX/LEN/CMD/PAYLOAD/CKSUM/ETX, matching host_gui/fgpt_gui/protocol.py.
#define FPGA_FRAMED     0

// Raw mode: finish a reply after this much silence, or this long overall.
#define FPGA_IDLE_MS    1500
#define FPGA_MAX_MS     20000

// Set to 1 to answer prompts locally (no FPGA wired), for UI testing.
#define FPGA_MOCK       0

// ── HTTP ──────────────────────────────────────────────────────────────
#define HTTP_PORT       80
// Maximum prompt length accepted by /api/generate.
#define MAX_PROMPT_LEN  256
