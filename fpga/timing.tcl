# timing.tcl — report Fmax and setup paths for fpGPT
#
# Usage (from fpga/, after build.tcl has produced a fitted design):
#     quartus_sta -t timing.tcl
#
# Area/resource usage is written by the fitter to output_files/fpGPT.fit.rpt.

package require ::quartus::sta

set script_dir [file dirname [file normalize [info script]]]
cd $script_dir

project_open fpGPT
create_timing_netlist
read_sdc
update_timing_netlist

puts "\n===== Clock Fmax ====="
report_clock_fmax_summary

puts "\n===== Worst setup paths ====="
report_timing -setup -npaths 10 -detail summary

project_close
