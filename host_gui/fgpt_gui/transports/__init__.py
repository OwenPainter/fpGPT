"""Transport implementations."""

from __future__ import annotations

from .base import Transport, TransportError

__all__ = ["Transport", "TransportError", "build_transport"]


def build_transport(config, params):
    """Construct the transport named by ``config``.

    Imports are local so that a missing optional dependency (pyserial for the
    serial transport) only matters when that transport is actually selected.
    """
    if config.transport == "serial":
        from .serial_transport import SerialTransport

        return SerialTransport(
            port=config.port,
            baud=config.baud or params.baud_rate,
        )

    from .mock_transport import MockTransport

    return MockTransport(
        gen_tokens=config.gen_tokens or params.gen_tokens,
        max_seq_len=params.max_seq_len,
        vocab_size=params.vocab_size,
        style=config.mock_style,
        corpus=config.corpus,
    )