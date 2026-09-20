# build.tcl — Quartus build flow for fpGPT
#
# Usage (from fpga/):
#     quartus_sh -t build.tcl
#
# It syncs the compiler-generated board parameters and ROM image into this
# directory, then runs analysis/synthesis, fitter, assembler and timing
# analysis. The bitstream lands in output_files/fpGPT.sof.

package require ::quartus::project
package require ::quartus::flow

set script_dir [file dirname [file normalize [info script]]]
cd $script_dir

# ── 1. Bring in compiler-generated artifacts ──
proc sync_generated {} {
    set params [file normalize "../build/rtl/board_params.vh"]
    if {[file exists $params]} {
        file copy -force $params "board_params.vh"
        puts "\[build\] board_params.vh <- $params"
    } else {
        puts "\[build\] WARNING: $params not found; using checked-in board_params.vh"
    }

    set rom [file normalize "../build/weights/engine/weights_unified.hex"]
    if {[file exists $rom]} {
        file copy -force $rom "weights_unified.hex"
        puts "\[build\] weights_unified.hex <- $rom"
    } else {
        puts "\[build\] WARNING: $rom not found; using an all-zero ROM image"
        write_zero_rom
    }
}

# Emit an all-zero ROM image sized to FPGPT_ROM_DEPTH so the MIF_FILE
# assignment and $readmemh always resolve. The model then emits '.' only.
proc write_zero_rom {} {
    set depth 0
    if {[file exists "board_params.vh"]} {
        set fh [open "board_params.vh" r]
        set text [read $fh]
        close $fh
        regexp {FPGPT_ROM_DEPTH\s*=\s*(\d+)} $text -> depth
    }
    if {$depth <= 0} {
        puts "\[build\] WARNING: could not read FPGPT_ROM_DEPTH; writing empty ROM"
        set depth 1
    }
    set fh [open "weights_unified.hex" w]
    for {set i 0} {$i < $depth} {incr i} { puts $fh "00" }
    close $fh
    puts "\[build\] weights_unified.hex <- all-zero image ($depth bytes)"
}
sync_generated

# ── 2. Compile ──
if {[catch {project_open fpGPT} err]} {
    puts "\[build\] ERROR opening project: $err"
    exit 1
}
execute_flow -compile
project_close

puts "\[build\] Done. Bitstream: [pwd]/output_files/fpGPT.sof"
puts "\[build\] For Fmax, run: quartus_sta -t timing.tcl"
