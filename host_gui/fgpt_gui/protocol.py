"""Framed UART protocol codec (host side).

Wire format (see docs/frontend_gui_plan.md section 5.2)::

    STX(0x02) | LEN(1) | CMD(1) | PAYLOAD[LEN] | CKSUM(1) | ETX(0x03)
    CKSUM = 8-bit XOR of LEN, CMD and PAYLOAD

The decoder is a streaming byte parser: feed it whatever chunks arrive and it
returns the complete, checksum-validated frames it can build. It resynchronises
on the next STX after any malformed frame, so line noise cannot wedge it.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

STX = 0x02
ETX = 0x03
MAX_PAYLOAD = 255


class Command(IntEnum):
    """Host -> FPGA and FPGA -> host frame commands."""

    # Host -> FPGA
    PING = 0x01
    PROMPT = 0x02
    GENERATE = 0x03
    RESET = 0x04
    INFO = 0x05

    # FPGA -> host
    READY = 0x81
    TOKEN = 0x82
    DONE = 0x83
    INFO_RESP = 0x84
    ERROR = 0x85


class DoneReason(IntEnum):
    OK = 0
    TIMEOUT = 1
    ERROR = 2
    OVERFLOW = 3


class FrameError(ValueError):
    """Raised when a frame cannot be encoded."""


def checksum(data: bytes) -> int:
    value = 0
    for byte in data:
        value ^= byte
    return value & 0xFF


@dataclass(frozen=True)
class Frame:
    command: int
    payload: bytes = b""

    @property
    def cmd(self) -> Command | None:
        try:
            return Command(self.command)
        except ValueError:
            return None


def encode_frame(command, payload: bytes = b"") -> bytes:
    """Encode one frame, validating the length and command byte."""
    payload = bytes(payload)
    if len(payload) > MAX_PAYLOAD:
        raise FrameError(f"payload too long: {len(payload)} > {MAX_PAYLOAD}")
    try:
        cmd_byte = int(command) & 0xFF
    except (TypeError, ValueError) as exc:
        raise FrameError(f"invalid command {command!r}") from exc

    body = bytes([len(payload), cmd_byte]) + payload
    return bytes([STX]) + body + bytes([checksum(body)]) + bytes([ETX])


class FrameDecoder:
    """Incremental frame decoder with resynchronisation."""

    def __init__(self) -> None:
        self._buf = bytearray()
        self.errors = 0

    def reset(self) -> None:
        self._buf.clear()

    def feed(self, data: bytes) -> list[Frame]:
        """Consume ``data`` and return every complete valid frame."""
        self._buf.extend(data)
        frames: list[Frame] = []

        while True:
            start = self._buf.find(STX)
            if start < 0:
                # No frame opener anywhere; drop the noise.
                if self._buf:
                    self.errors += 1
                    self._buf.clear()
                return frames
            if start > 0:
                # Discarded bytes before an opener.
                self.errors += 1
                del self._buf[:start]

            # STX + LEN + CMD + CKSUM + ETX is the smallest possible frame.
            if len(self._buf) < 5:
                return frames

            length = self._buf[1]
            total = 5 + length
            if len(self._buf) < total:
                return frames

            if self._buf[total - 1] != ETX:
                self.errors += 1
                del self._buf[:1]  # skip this STX, try the next one
                continue

            body = bytes(self._buf[1:3 + length])
            received = self._buf[3 + length]
            if checksum(body) != received:
                self.errors += 1
                del self._buf[:1]
                continue

            payload = bytes(self._buf[3:3 + length])
            frames.append(Frame(self._buf[2], payload))
            del self._buf[:total]

        return frames