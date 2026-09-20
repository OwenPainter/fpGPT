# build.tcl — Quartus build flow for fpGPT
#
# Usage (from fpga/):
#     quartus_sh -t build.tcl
#
# It syncs the compiler-generated board parameters and ROM image into this
# directory, then runs analysis/synthesis, fitter, assembler and timing
# analysis. The bitstream lands in output_files/fpGPT.sof.

set script_dir [file dirname [file normalize [info script]]]
cd $script_dir

# ── 1. Bring in compiler-generated artifacts ──
proc sync_generated {} {
    set params [file normalize "../build/rtl/board_params.vh"]
    set rom [file normalize "../build/weights/engine/weights_unified.hex"]
    foreach path [list $params $rom] {
        if {![file exists $path]} {
            error "Missing $path; run compile.py with a trained checkpoint first"
        }
    }
    set fh [open $params r]
    set header [read $fh]
    close $fh
    foreach name {ROM_DEPTH DATA_WIDTH MAX_SEQ_LEN GEN_TOKENS} {
        if {![regexp "FPGPT_${name}\\s+(\\d+)" $header -> value($name)]} {
            error "Missing FPGPT_$name in $params"
        }
    }
    if {$value(GEN_TOKENS) < 1 || $value(GEN_TOKENS) >= $value(MAX_SEQ_LEN)} {
        error "GEN_TOKENS must leave at least one prompt slot"
    }
    set fh [open $rom r]
    set words [regexp -all -inline {\S+} [read $fh]]
    close $fh
    if {[llength $words] != $value(ROM_DEPTH)} {
        error "ROM contains [llength $words] words; expected $value(ROM_DEPTH)"
    }
    set digits [expr {($value(DATA_WIDTH) + 3) / 4}]
    foreach word $words {
        if {![regexp {^[0-9a-fA-F]+$} $word] || [string length $word] != $digits} {
            error "Invalid ROM word '$word'; expected $digits hexadecimal digits"
        }
    }
    file copy -force $params "board_params.vh"
    file copy -force $rom "weights_unified.hex"
    # Bind a MIF directly to the inferred ROM. Expanding a 212K-word
    # $readmemh initial block in Quartus 25.1 is prohibitively slow.
    # Derive it from the validated simulation image so both paths agree.
    set fh [open "weights_unified.hex.mif" w]
    puts $fh "WIDTH=$value(DATA_WIDTH);"
    puts $fh "DEPTH=$value(ROM_DEPTH);"
    puts $fh "ADDRESS_RADIX=HEX;"
    puts $fh "DATA_RADIX=HEX;"
    puts $fh "CONTENT BEGIN"
    set address 0
    foreach word $words {
        puts $fh [format "%X : %s;" $address $word]
        incr address
    }
    puts $fh "END;"
    close $fh
    puts "\[build\] Synced validated parameters and [llength $words] ROM words"
}
sync_generated

# Validate/sync artifacts without requiring Quartus (also used by tests).
if {[info exists ::env(FPGPT_PREPARE_ONLY)] && $::env(FPGPT_PREPARE_ONLY) eq "1"} {
    exit 0
}

package require ::quartus::project
package require ::quartus::flow

# ── 2. Compile ──
if {[catch {project_open fpGPT} err]} {
    puts "\[build\] ERROR opening project: $err"
    exit 1
}
execute_flow -compile
project_close

puts "\[build\] Done. Bitstream: [pwd]/output_files/fpGPT.sof"
puts "\[build\] For Fmax, run: quartus_sta -t timing.tcl"
