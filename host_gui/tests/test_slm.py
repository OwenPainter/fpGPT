import time
import pytest
from fgpt_gui.config import GuiConfig
from fgpt_gui.transports import build_transport
from fgpt_gui.transports.slm_transport import SlmTransport
from fgpt_gui.board_params import BoardParams


def test_slm_transport_lifecycle():
    t = SlmTransport(gen_tokens=16, byte_delay=0.001)
    t.open()
    t.send(b"Hello world\n")
    time.sleep(0.3)
    data = t.read(1.0)
    assert len(data) > 0
    t.reset()
    t.close()


def test_build_slm_transport():
    cfg = GuiConfig(transport="slm", slm_engine="microgpt", gen_tokens=16)
    params = BoardParams(gen_tokens=16, max_seq_len=64, vocab_size=64, d_model=64,
                         num_layers=4, num_heads=4, d_ff=256, w_addr_width=18,
                         rom_depth=212352, baud_rate=115200, sys_clk_hz=62500000)
    transport = build_transport(cfg, params)
    assert isinstance(transport, SlmTransport)


def test_slm_session_natural_conversation():
    from fgpt_gui.session import ChatSession, SessionState

    params = BoardParams(gen_tokens=16, max_seq_len=64, vocab_size=64, d_model=64,
                         num_layers=4, num_heads=4, d_ff=256, w_addr_width=18,
                         rom_depth=212352, baud_rate=115200, sys_clk_hz=62500000)
    t = SlmTransport(engine="assistant", byte_delay=0.0005)
    session = ChatSession(t, params, mode="legacy", gen_tokens=256, reply_timeout=3.0)

    clean, removed = session.sanitize("Hello! How are you?")
    assert len(removed) == 0
    assert clean == "Hello! How are you?"

    with session:
        sub = session.subscribe()
        ok, err = session.submit(clean)
        assert ok is True

        reply_done = False
        reply_text = ""
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            event = sub.get(timeout=1.0)
            if event.get("type") == "reply_end":
                reply_done = True
                reply_text = event.get("text", "")
                break

        assert reply_done is True
        assert len(reply_text) > 10
        assert not reply_text.startswith(" ")
        assert "hello" in reply_text.lower() or "knowledge base" in reply_text.lower() or "how can i help" in reply_text.lower()
        assert session.state == SessionState.IDLE

