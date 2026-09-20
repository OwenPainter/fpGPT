// ESP32-S2 USB <-> UART bridge
//
// Makes the board behave like a plain USB-to-UART adapter: anything typed or
// sent on the native USB serial port is forwarded out the hardware UART, and
// anything arriving on the UART is forwarded back to USB. Binary-safe, no line
// buffering, no prompts or banners in the data path.
//
//   Computer  --USB CDC-->  ESP32-S2  --UART TX/RX-->  target device
//
// Arduino IDE setup (Tools menu):
//   Board            : ESP32S2 Dev Module
//   USB CDC On Boot  : Enabled     <-- required, so Serial == USB
//   Upload Speed     : 921600
//
// UART settings and pins live in config.h.

#include "config.h"

static uint8_t pumpBuf[BRIDGE_CHUNK];
static uint32_t lastRxMs = 0;
static bool ledOn = false;
#if FOLLOW_USB_BAUD
static uint32_t uartBaud = UART_BAUD;
static uint32_t lastBaudCheck = 0;
#endif

// Move everything currently available from `from` to `to`.
static void pump(Stream& from, Stream& to) {
  while (from.available() > 0) {
    size_t want = from.available();
    if (want > sizeof(pumpBuf)) want = sizeof(pumpBuf);
    int n = from.read(pumpBuf, want);
    if (n <= 0) break;
    to.write(pumpBuf, n);
    lastRxMs = millis();
  }
}

// Discard a stale half-frame when the link reconnects.
static void drain(Stream& s) {
  while (s.available() > 0) s.read();
}

#if FOLLOW_USB_BAUD
static void followUsbBaud() {
  if (millis() - lastBaudCheck < 250) return;
  lastBaudCheck = millis();
  uint32_t baud = Serial.baudRate();
  if (baud >= 300 && baud != uartBaud) {
    uartBaud = baud;
    Serial1.updateBaudRate(baud);
    Serial1.flush();
  }
}
#endif

static void statusLed() {
#if LED_PIN >= 0
  bool active = (millis() - lastRxMs) < LED_IDLE_MS;
  if (active != ledOn) {
    ledOn = active;
    digitalWrite(LED_PIN, LED_ACTIVE_HIGH ? active : !active);
  }
#endif
}

void setup() {
  Serial.begin(USB_BAUD);  // native USB CDC (host port)

  Serial1.setRxBufferSize(UART_RX_BUFFER);
  Serial1.setTxBufferSize(UART_TX_BUFFER);
  Serial1.begin(UART_BAUD, SERIAL_8N1, UART_RX_PIN, UART_TX_PIN);

#if LED_PIN >= 0
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LED_ACTIVE_HIGH ? LOW : HIGH);
#endif

  drain(Serial);
  drain(Serial1);
}

void loop() {
  // Transparent bidirectional copy.
  pump(Serial, Serial1);   // host -> device
  pump(Serial1, Serial);   // device -> host

#if FOLLOW_USB_BAUD
  followUsbBaud();
#endif

  statusLed();

  // USB CDC can re-enumerate when the host connects; reset internal state.
  if (!Serial) {
    Serial1.flush();
    delay(1);
  }
}