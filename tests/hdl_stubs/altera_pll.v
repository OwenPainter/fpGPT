// ═══════════════════════════════════════════════════════════════
// altera_pll — blackbox stub for lint / elaboration only
//
// fpga/pll_150.v instantiates the Quartus `altera_pll` megafunction, which
// only exists inside Quartus. This file provides an empty, port-compatible
// module so Icarus Verilog can elaborate pll_150 and Verilator can lint it.
// It is NOT part of the synthesizable design and is never compiled by Quartus.
// ═══════════════════════════════════════════════════════════════
`timescale 1ns/1ps

/* verilator lint_off UNUSEDPARAM */
/* verilator lint_off UNUSEDSIGNAL */
/* verilator lint_off UNDRIVEN */
module altera_pll #(
    parameter fractional_vco_multiplier = "false",
    parameter reference_clock_frequency = "0 MHz",
    parameter operation_mode = "direct",
    parameter number_of_clocks = 1,
    parameter output_clock_frequency0 = "0 MHz",
    parameter output_clock_frequency1 = "0 MHz",
    parameter output_clock_frequency2 = "0 MHz",
    parameter output_clock_frequency3 = "0 MHz",
    parameter output_clock_frequency4 = "0 MHz",
    parameter output_clock_frequency5 = "0 MHz",
    parameter output_clock_frequency6 = "0 MHz",
    parameter output_clock_frequency7 = "0 MHz",
    parameter output_clock_frequency8 = "0 MHz",
    parameter output_clock_frequency9 = "0 MHz",
    parameter output_clock_frequency10 = "0 MHz",
    parameter output_clock_frequency11 = "0 MHz",
    parameter output_clock_frequency12 = "0 MHz",
    parameter output_clock_frequency13 = "0 MHz",
    parameter output_clock_frequency14 = "0 MHz",
    parameter output_clock_frequency15 = "0 MHz",
    parameter output_clock_frequency16 = "0 MHz",
    parameter output_clock_frequency17 = "0 MHz",
    parameter phase_shift0 = "0 ps",
    parameter phase_shift1 = "0 ps",
    parameter phase_shift2 = "0 ps",
    parameter phase_shift3 = "0 ps",
    parameter phase_shift4 = "0 ps",
    parameter phase_shift5 = "0 ps",
    parameter phase_shift6 = "0 ps",
    parameter phase_shift7 = "0 ps",
    parameter phase_shift8 = "0 ps",
    parameter phase_shift9 = "0 ps",
    parameter phase_shift10 = "0 ps",
    parameter phase_shift11 = "0 ps",
    parameter phase_shift12 = "0 ps",
    parameter phase_shift13 = "0 ps",
    parameter phase_shift14 = "0 ps",
    parameter phase_shift15 = "0 ps",
    parameter phase_shift16 = "0 ps",
    parameter phase_shift17 = "0 ps",
    parameter duty_cycle0 = 50,
    parameter duty_cycle1 = 50,
    parameter duty_cycle2 = 50,
    parameter duty_cycle3 = 50,
    parameter duty_cycle4 = 50,
    parameter duty_cycle5 = 50,
    parameter duty_cycle6 = 50,
    parameter duty_cycle7 = 50,
    parameter duty_cycle8 = 50,
    parameter duty_cycle9 = 50,
    parameter duty_cycle10 = 50,
    parameter duty_cycle11 = 50,
    parameter duty_cycle12 = 50,
    parameter duty_cycle13 = 50,
    parameter duty_cycle14 = 50,
    parameter duty_cycle15 = 50,
    parameter duty_cycle16 = 50,
    parameter duty_cycle17 = 50,
    parameter pll_type = "General",
    parameter pll_subtype = "General"
) (
    input  wire       rst,
    output wire [0:0] outclk,
    output wire       locked,
    output wire       fboutclk,
    input  wire       fbclk,
    input  wire       refclk
);
endmodule
/* verilator lint_on UNUSEDPARAM */
/* verilator lint_on UNUSEDSIGNAL */
/* verilator lint_on UNDRIVEN */
