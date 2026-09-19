"""pyserial transport for a real DE1-SoC UART link.

The board's UART is on ``GPIO_0[0:1]`` via an external 3.3 V USB-TTL adapter;
see ``fpga/PROGRAMMING.md``. Settings are 115200 8N1, no flow control.
"""

from __future__ import annotations

import time

from .base import Transport, TransportError


class SerialTransport(Transport):
    name = "serial"

    def __init__(self, port: str, baud: int = 115200, read_chunk: int = 256) -> None:
        self.port = port
        self.baud = int(baud)
        self.read_chunk = read_chunk
        self._serial = None

    def open(self) -> None:
        try:
            import serial  # pyserial
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise TransportError(
                "pyserial is required for the serial transport: "
                "pip install pyserial"
            ) from exc

        try:
            self._serial = serial.Serial(
                self.port, self.baud, timeout=0.0, write_timeout=1.0
            )
        except Exception as exc:  # pyserial raises SerialException
            raise TransportError(f"could not open {self.port}: {exc}") from exc

    def close(self) -> None:
        if self._serial is not None:
            try:
                self._serial.close()
            finally:
                self._serial = None

    def send(self, data: bytes) -> None:
        if self._serial is None:
            raise TransportError("serial port is not open")
        try:
            self._serial.write(data)
            self._serial.flush()
        except Exception as exc:
            raise TransportError(f"serial write failed: {exc}") from exc

    def read(self, timeout: float) -> bytes:
        if self._serial is None:
            raise TransportError("serial port is not open")
        deadline = time.monotonic() + max(0.0, timeout)
        buf = bytearray()
        try:
            while True:
                waiting = self._serial.in_waiting
                if waiting:
                    buf.extend(self._serial.read(min(waiting, self.read_chunk)))
                    if len(buf) >= self.read_chunk:
                        break
                if buf and not waiting:
                    break
                if time.monotonic() >= deadline:
                    break
                time.sleep(0.002)
        except Exception as exc:
            raise TransportError(f"serial read failed: {exc}") from exc
        return bytes(buf)

    @staticmethod
    def list_ports() -> list[str]:
        try:
            from serial.tools import list_ports

            return [p.device for p in list_ports.comports()]
        except Exception:  # pragma: no cover
            return []