"""Host GUI configuration.

Kept deliberately small and dependency-free so the CLI and tests can construct
it anywhere. Runtime defaults are filled in from ``board_params.vh`` by the
server/CLI instead of being duplicated here.
"""

from __future__ import annotations

from dataclasses import dataclass


VALID_TRANSPORTS = ("mock", "serial")
VALID_MODES = ("legacy", "framed")
VALID_MOCK_STYLES = ("echo", "ngram")


@dataclass
class GuiConfig:
    """Everything the host bridge needs to start."""

    # Link to the board.
    transport: str = "mock"
    mode: str = "legacy"

    # Serial transport options (ignored by the mock).
    port: str | None = None
    baud: int | None = None

    # Generation options.
    gen_tokens: int | None = None
    timeout_s: float = 5.0

    # Where to find the compiler-generated parameter contract.
    board_params: str | None = None

    # TFLite model for the "food photo" tab (hot-dog classifier).
    # ``None`` uses the bundled Models/hotdog_model.tflite.
    food_model: str | None = None

    # Mock transport options.
    mock_style: str = "echo"
    corpus: str | None = None

    # Local HTTP server.
    http_host: str = "127.0.0.1"
    http_port: int = 8765

    def validate(self) -> None:
        if self.transport not in VALID_TRANSPORTS:
            raise ValueError(
                f"transport must be one of {VALID_TRANSPORTS}, got {self.transport!r}"
            )
        if self.mode not in VALID_MODES:
            raise ValueError(f"mode must be one of {VALID_MODES}, got {self.mode!r}")
        if self.mock_style not in VALID_MOCK_STYLES:
            raise ValueError(
                f"mock_style must be one of {VALID_MOCK_STYLES}, "
                f"got {self.mock_style!r}"
            )
        if self.transport == "serial" and not self.port:
            raise ValueError("transport=serial requires --port")
        if self.gen_tokens is not None and not (1 <= self.gen_tokens <= 255):
            raise ValueError("gen_tokens must be in 1..255")
        if self.timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        if not (0 <= self.http_port <= 65535):
            raise ValueError("http_port out of range")