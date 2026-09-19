// fpGPT ESP32-S2 webserver — UART link to the FPGA.
//
// Two link modes, selected by FPGA_FRAMED in config.h:
//   * raw    — prompt bytes + CR/LF, stream reply bytes (current board firmware)
//   * framed — STX/LEN/CMD/PAYLOAD/CKSUM/ETX, matching
//              host_gui/fgpt_gui/protocol.py
//
// The framed decoder resynchronises on the next STX after a bad frame, so line
// noise cannot wedge the link.

#pragma once

#include <Arduino.h>

#include "config.h"

namespace fpgpt {

constexpr uint8_t STX = 0x02;
constexpr uint8_t ETX = 0x03;
constexpr uint8_t MAX_PAYLOAD = 255;

// Command bytes (host -> FPGA and FPGA -> host).
enum Cmd : uint8_t {
  PING = 0x01,
  PROMPT = 0x02,
  GENERATE = 0x03,
  RESET = 0x04,
  INFO = 0x05,
  READY = 0x81,
  TOKEN = 0x82,
  DONE = 0x83,
  INFO_RESP = 0x84,
  ERROR = 0x85,
};

struct Frame {
  uint8_t cmd = 0;
  uint8_t payload[MAX_PAYLOAD];
  size_t len = 0;
};

// Encode one frame into `out`; returns the byte count, or 0 if it will not fit.
inline size_t encodeFrame(uint8_t cmd, const uint8_t* payload, size_t len,
                          uint8_t* out, size_t outCap) {
  if (len > MAX_PAYLOAD || outCap < len + 5) return 0;
  out[0] = STX;
  out[1] = static_cast<uint8_t>(len);
  out[2] = cmd;
  uint8_t checksum = static_cast<uint8_t>(len) ^ cmd;
  for (size_t i = 0; i < len; ++i) {
    out[3 + i] = payload[i];
    checksum ^= payload[i];
  }
  out[3 + len] = checksum;
  out[4 + len] = ETX;
  return len + 5;
}

// Streaming frame decoder: feed it bytes, get complete validated frames.
class FrameDecoder {
 public:
  void reset() {
    state_ = WAIT_STX;
    pos_ = 0;
  }

  uint32_t errors() const { return errors_; }

  bool feed(uint8_t byte, Frame& out) {
    switch (state_) {
      case WAIT_STX:
        if (byte == STX) {
          state_ = LEN;
        } else if (byte != 0) {
          ++errors_;
        }
        break;

      case LEN:
        len_ = byte;
        pos_ = 0;
        state_ = (byte <= MAX_PAYLOAD) ? CMD : WAIT_STX;
        break;

      case CMD:
        cmd_ = byte;
        state_ = (len_ == 0) ? CKSUM : PAYLOAD;
        break;

      case PAYLOAD:
        payload_[pos_++] = byte;
        if (pos_ >= len_) state_ = CKSUM;
        break;

      case CKSUM: {
        uint8_t checksum = static_cast<uint8_t>(len_) ^ cmd_;
        for (size_t i = 0; i < len_; ++i) checksum ^= payload_[i];
        if (checksum != byte) {
          ++errors_;
          state_ = WAIT_STX;
        } else {
          state_ = END;
        }
        break;
      }

      case END:
        state_ = WAIT_STX;
        if (byte != ETX) {
          ++errors_;
          break;
        }
        out.cmd = cmd_;
        out.len = len_;
        for (size_t i = 0; i < len_; ++i) out.payload[i] = payload_[i];
        return true;
    }
    return false;
  }

 private:
  enum State { WAIT_STX, LEN, CMD, PAYLOAD, CKSUM, END };
  State state_ = WAIT_STX;
  uint8_t len_ = 0;
  uint8_t cmd_ = 0;
  uint8_t payload_[MAX_PAYLOAD];
  size_t pos_ = 0;
  uint32_t errors_ = 0;
};

// Thin wrapper around Serial1.
class FpgaLink {
 public:
  void begin() {
    Serial1.begin(FPGA_BAUD, SERIAL_8N1, FPGA_RX_PIN, FPGA_TX_PIN);
  }

  // Raw mode: send the prompt followed by CR/LF (the firmware triggers on it).
  void sendRawPrompt(const String& prompt) {
    Serial1.write(reinterpret_cast<const uint8_t*>(prompt.c_str()), prompt.length());
    Serial1.write('\n');
    Serial1.flush();
  }

  // Framed mode.
  bool sendFrame(uint8_t cmd, const uint8_t* payload, size_t len) {
    uint8_t buffer[MAX_PAYLOAD + 5];
    size_t n = encodeFrame(cmd, payload, len, buffer, sizeof(buffer));
    if (n == 0) return false;
    Serial1.write(buffer, n);
    Serial1.flush();
    return true;
  }

  bool sendPrompt(const String& prompt) {
    return sendFrame(PROMPT, reinterpret_cast<const uint8_t*>(prompt.c_str()),
                     prompt.length());
  }

  bool sendGenerate(uint8_t count) {
    uint8_t payload[1] = {count};
    return sendFrame(GENERATE, payload, 1);
  }

  bool sendReset() { return sendFrame(RESET, nullptr, 0); }

  int available() { return Serial1.available(); }
  int read() { return Serial1.read(); }

  void flushInput() {
    while (Serial1.available()) Serial1.read();
  }

  FrameDecoder& decoder() { return decoder_; }

 private:
  FrameDecoder decoder_;
};

}  // namespace fpgpt
