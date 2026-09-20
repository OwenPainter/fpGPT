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
