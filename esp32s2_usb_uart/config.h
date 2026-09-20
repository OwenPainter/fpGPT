// ESP32-S2 USB <-> UART bridge — configuration.
//
// Edit before flashing. Only the pins/baud usually need changing.

#pragma once

// ── UART side (wired to the target device) ────────────────────────────
//   ESP32 UART_RX_PIN  <- target TX
//   ESP32 UART_TX_PIN  -> target RX
//   GND                <-> GND
// 3.3 V logic only. ESP32-S2 can route UART1 to almost any GPIO.
#define UART_BAUD        115200
#define UART_RX_PIN      18
#define UART_TX_PIN      17
#define UART_RX_BUFFER   1024
#define UART_TX_BUFFER   1024

// ── USB side (native USB-C / Micro-USB port to your computer) ─────────
// Only affects the host-side line speed hint; CDC ignores it in practice.
#define USB_BAUD         115200

// 1 = adopt the baud rate selected in the host serial monitor and apply it
//     to the UART side. Useful as a real USB-UART dongle. 0 = use UART_BAUD.
#define FOLLOW_USB_BAUD  0

// ── Bridge behaviour ──────────────────────────────────────────────────
#define BRIDGE_CHUNK     256   // max bytes moved per pump
#define LED_PIN          -1    // optional activity LED (-1 = none)
#define LED_ACTIVE_HIGH  1
#define LED_IDLE_MS      40