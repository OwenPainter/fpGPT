// ═══════════════════════════════════════════════════════════════════════
// fpGPT — ESP32-S2 WiFi webserver / FPGA bridge
//
// Flash with the Arduino IDE (see README.md). The board joins WiFi, serves a
// chat page, and forwards prompts to the DE1-SoC FPGA over UART, streaming the
// generated characters back to the browser.
//
//   Browser  --HTTP-->  ESP32-S2  --UART 115200-->  FPGA (fpga_top.v)
//
// Endpoints:
//   GET  /               chat page
//   GET  /api/status     JSON link/health
//   POST /api/generate   body = prompt; streamed text/plain reply
//   POST /api/reset      reset/flush the FPGA link
//
// Edit config.h for WiFi credentials, UART pins and link mode.
// ═══════════════════════════════════════════════════════════════════════

#include <WiFi.h>
#include <WebServer.h>

#include "config.h"
#include "fpga_link.h"
#include "web_page.h"

WebServer server(HTTP_PORT);
fpgpt::FpgaLink fpga;

// ── WiFi ──────────────────────────────────────────────────────────────
static void connectWifi() {
  WiFi.mode(WIFI_STA);
  WiFi.setHostname(WIFI_HOSTNAME);
  WiFi.setSleep(false);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  Serial.print("[wifi] connecting to ");
  Serial.print(WIFI_SSID);
  unsigned long start = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - start < WIFI_TIMEOUT_MS) {
    delay(400);
    Serial.print('.');
  }
  Serial.println();
  if (WiFi.status() == WL_CONNECTED) {
    Serial.print("[wifi] connected, IP ");
    Serial.println(WiFi.localIP());
  } else {
    Serial.println("[wifi] not connected; server still starting");
  }
}

// ── Handlers ──────────────────────────────────────────────────────────
static void handleRoot() {
  server.send_P(200, "text/html", INDEX_HTML);
}

static void handleStatus() {
  String json = "{";
  json += "\"wifi\":";
  json += (WiFi.status() == WL_CONNECTED) ? "true" : "false";
  json += ",\"ip\":\"" + WiFi.localIP().toString() + "\"";
  json += ",\"rssi\":" + String(WiFi.status() == WL_CONNECTED ? WiFi.RSSI() : 0);
  json += ",\"baud\":" + String(FPGA_BAUD);
  json += ",\"framed\":" + String(FPGA_FRAMED);
  json += ",\"mock\":" + String(FPGA_MOCK);
  json += ",\"decoder_errors\":" + String(fpga.decoder().errors());
  json += "}";
  server.send(200, "application/json", json);
}

static void handleReset() {
#if FPGA_FRAMED
  fpga.sendReset();
#endif
  fpga.flushInput();
  server.send(200, "text/plain; charset=utf-8", "ok");
}

#if FPGA_MOCK
static String mockReply(const String& prompt) {
  String reply = " (mock) I heard: \"" + prompt + "\". ";
  reply += "Wire the FPGA UART and set FPGA_MOCK 0 for real output.";
  return reply;
}
#endif

// Streams the FPGA reply as a chunked text/plain response.
static void handleGenerate() {
  String prompt = server.arg("plain");
  prompt.trim();
  if (prompt.length() == 0) {
    server.send(400, "text/plain; charset=utf-8", "empty prompt");
    return;
  }
  if (prompt.length() > MAX_PROMPT_LEN) {
    prompt = prompt.substring(0, MAX_PROMPT_LEN);
  }

  Serial.print("[gen] prompt: ");
  Serial.println(prompt);

  server.setContentLength(CONTENT_LENGTH_UNKNOWN);
  server.send(200, "text/plain; charset=utf-8", "");

#if FPGA_MOCK
  server.sendContent(mockReply(prompt));
  server.sendContent("");
  return;
#endif

  fpga.flushInput();
  const unsigned long start = millis();
  String chunk;
  chunk.reserve(64);

#if FPGA_FRAMED
  fpga.sendPrompt(prompt);
  fpga.sendGenerate(0);  // 0 = firmware default token count
  fpgpt::Frame frame;
  while (millis() - start < FPGA_MAX_MS) {
    int byte = fpga.read();
    if (byte < 0) {
      delay(1);
      continue;
    }
    if (!fpga.decoder().feed(static_cast<uint8_t>(byte), frame)) continue;
    if (frame.cmd == fpgpt::TOKEN) {
      for (size_t i = 0; i < frame.len; ++i) chunk += static_cast<char>(frame.payload[i]);
      if (chunk.length() >= 64) {
        server.sendContent(chunk);
        chunk = "";
      }
    } else if (frame.cmd == fpgpt::DONE) {
      break;
    } else if (frame.cmd == fpgpt::ERROR) {
      chunk += "\n[fpga error]";
      break;
    }
  }
#else
  // Raw mode: the firmware replies with raw bytes and no terminator, so stop
  // after a quiet gap or an overall deadline.
  fpga.sendRawPrompt(prompt);
  unsigned long last = start;
  while (millis() - start < FPGA_MAX_MS) {
    int byte = fpga.read();
    if (byte >= 0) {
      chunk += static_cast<char>(byte);
      last = millis();
      if (chunk.length() >= 64) {
        server.sendContent(chunk);
        chunk = "";
      }
    } else if (millis() - last > FPGA_IDLE_MS) {
      break;
    } else {
      delay(1);
    }
  }
#endif

  if (chunk.length()) server.sendContent(chunk);
  server.sendContent("");  // terminate the chunked response
}

static void handleNotFound() {
  server.send(404, "text/plain; charset=utf-8", "not found");
}

// ── Arduino entry points ──────────────────────────────────────────────
void setup() {
  Serial.begin(115200);
  delay(200);
  Serial.println();
  Serial.println("[boot] fpGPT ESP32-S2 webserver");

  fpga.begin();

  connectWifi();

  server.on("/", HTTP_GET, handleRoot);
  server.on("/api/status", HTTP_GET, handleStatus);
  server.on("/api/generate", HTTP_POST, handleGenerate);
  server.on("/api/reset", HTTP_POST, handleReset);
  server.onNotFound(handleNotFound);
  server.begin();
  Serial.println("[boot] http server started");
}

void loop() {
  server.handleClient();
}
