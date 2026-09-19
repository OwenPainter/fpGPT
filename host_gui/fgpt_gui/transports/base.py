"""Transport layer: how the host talks to the board (or a mock).

A transport is intentionally boring and synchronous: ``send`` writes bytes,
``read`` returns whatever arrived within a timeout (``b''`` means "nothing yet",
never "closed"). The session runs the reader loop on a background thread.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class TransportError(RuntimeError):
    """Raised when the underlying link fails (e.g. the serial port vanishes)."""


class Transport(ABC):
    """Byte-oriented link to the board."""

    #: Human-readable label used in the UI/status events.
    name = "transport"

    @abstractmethod
    def open(self) -> None:
        """Acquire the link. Raises :class:`TransportError` on failure."""

    @abstractmethod
    def close(self) -> None:
        """Release the link. Must be safe to call more than once."""

    @abstractmethod
    def send(self, data: bytes) -> None:
        """Write ``data`` to the board."""

    @abstractmethod
    def read(self, timeout: float) -> bytes:
        """Return bytes received within ``timeout`` seconds (possibly ``b''``)."""

    def reset(self) -> None:
        """Optional board reset (no-op unless the transport supports it)."""