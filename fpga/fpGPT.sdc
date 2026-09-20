# ═══════════════════════════════════════════════════════════════
# fpGPT timing constraints — DE1-SoC / Cyclone V
#
# Clocks:
#   CLOCK_50  : 50 MHz board oscillator (PIN_AF14), 20 ns
#   clk_sys   : PLL output, 50 MHz * 5/4 = 62.5 MHz (16 ns)
#
# The operating point is set to the timing-closed Fmax (~73.5 MHz measured).
#
# The PLL output clock is derived automatically from the instantiated
# altera_pll so its net name is not hard-coded (the manual create_generated_clock
# never matched after synthesis).
# ═══════════════════════════════════════════════════════════════

# Base oscillator
create_clock -name CLOCK_50 -period 20.000 [get_ports {CLOCK_50}]

# Derive clk_sys (and any other PLL outputs) from the altera_pll instance
derive_pll_clocks

# ── Asynchronous inputs ──
# KEY[*] reset and UART_RXD are asynchronous to clk_sys. UART_RXD is captured
# by the two-flop synchronizer in uart_rx.v, so it is a false path.
set_false_path -from [get_ports {KEY[*]}]
set_false_path -from [get_ports {UART_RXD}]

# ── LED outputs ──
set_false_path -to [get_ports {LEDR[*]}]

# Derive on-chip clock uncertainty for the PLL and registers
derive_clock_uncertainty
