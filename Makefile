# fpGPT — one-command verification.
#
#   make test             Python unit tests + HDL elaboration/lint + board smoke test
#   make test-python      Python unit tests only
#   make test-hdl         Icarus elaboration + Verilator lint gate
#   make test-board       DE1-SoC board smoke test in Icarus Verilog
#   make lint             alias for test-hdl
#   make lint-baseline    regenerate the Verilator warning baseline
#   make golden-baseline  regenerate compiler golden vectors
#   make clean

PYTHON    ?= python3
PYTEST    ?= $(PYTHON) -m pytest
IVERILOG  ?= iverilog
VVP       ?= vvp
VERILATOR ?= verilator

BUILD_DIR := build/test_rtl
HDL_STUB  := tests/hdl_stubs/altera_pll.v
HDL_SRC   := $(wildcard hdl/*.v) $(wildcard fpga/*.v)
BOARD_TB  := tests/tb_fpga_top.v
BOARD_TOP := tb_fpga_top

.PHONY: test test-python test-hdl test-board lint lint-baseline golden-baseline clean

test: test-python test-hdl test-board

test-python:
	$(PYTEST) tests --ignore=tests/test_hdl_lint.py

test-hdl:
	$(PYTEST) tests/test_hdl_lint.py

test-board:
	@if ! command -v $(IVERILOG) >/dev/null 2>&1 || ! command -v $(VVP) >/dev/null 2>&1; then \
		echo "SKIP test-board: iverilog/vvp not found on PATH"; \
	else \
		mkdir -p $(BUILD_DIR); \
		$(IVERILOG) -g2012 -s $(BOARD_TOP) -Ifpga -Ihdl -o $(BUILD_DIR)/$(BOARD_TOP) \
			$(HDL_STUB) $(HDL_SRC) $(BOARD_TB) && \
		$(VVP) $(BUILD_DIR)/$(BOARD_TOP); \
	fi

lint: test-hdl

lint-baseline:
	FPGPT_UPDATE_LINT_BASELINE=1 $(PYTEST) tests/test_hdl_lint.py

golden-baseline:
	FPGPT_UPDATE_GOLDEN=1 $(PYTEST) tests/test_golden_vectors.py

clean:
	rm -rf $(BUILD_DIR) obj_dir
