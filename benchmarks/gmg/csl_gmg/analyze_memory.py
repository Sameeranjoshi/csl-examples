#!/usr/bin/env python3
"""
Memory Usage Analysis Script
Compares expected memory usage from CSL code with actual ELF output
"""

import sys

# Parameters from the compilation
MAX_ZDIM = 256
LEVELS = 9
BLOCK_SIZE = 256
TSC_SIZE_WORDS = 3  # From timestamp.tsc_size_words
TIMER_COUNT = 12  # From timer module

def compute_total_gmg_size(max_zdim, levels):
    """Compute geometric series: zdim + zdim/2 + zdim/4 + ... for LEVELS terms"""
    total = 0
    for level in range(levels):
        total += max_zdim >> level
    return total

def analyze_memory():
    print("=" * 70)
    print("MEMORY USAGE ANALYSIS")
    print("=" * 70)
    print(f"Parameters: MAX_ZDIM={MAX_ZDIM}, LEVELS={LEVELS}, BLOCK_SIZE={BLOCK_SIZE}")
    print()
    
    # Calculate TOTAL_GMG_SIZE
    TOTAL_GMG_SIZE = compute_total_gmg_size(MAX_ZDIM, LEVELS)
    print(f"TOTAL_GMG_SIZE (geometric series): {TOTAL_GMG_SIZE}")
    print(f"  Breakdown: {' + '.join([str(MAX_ZDIM >> i) for i in range(LEVELS)])}")
    print()
    
    # Main data arrays (f32 = 4 bytes)
    print("=" * 70)
    print("MAIN DATA ARRAYS (f32 = 4 bytes each)")
    print("=" * 70)
    main_arrays = {
        "u": MAX_ZDIM * 4,
        "f": MAX_ZDIM * 4,
        "r": MAX_ZDIM * 4,
        "Au": MAX_ZDIM * 4,
        "residual_temp": MAX_ZDIM * 4,
    }
    for name, size in main_arrays.items():
        print(f"  {name:20s}: {size:6d} bytes ({size/1024:6.2f} KB)")
    
    # GMG save arrays
    print()
    print("GMG SAVE ARRAYS (f32 = 4 bytes each)")
    print("-" * 70)
    gmg_arrays = {
        "save_u_smooth": TOTAL_GMG_SIZE * 4,
        "save_f_down": TOTAL_GMG_SIZE * 4,
    }
    for name, size in gmg_arrays.items():
        print(f"  {name:20s}: {size:6d} bytes ({size/1024:6.2f} KB)")
    
    # Level parameter arrays
    print()
    print("LEVEL PARAMETER ARRAYS (f32 = 4 bytes each)")
    print("-" * 70)
    level_arrays = {
        "hx_array": LEVELS * 4,
        "hy_array": LEVELS * 4,
        "hz_array": LEVELS * 4,
        "jacobi_coeff_array": LEVELS * 4,
    }
    for name, size in level_arrays.items():
        print(f"  {name:20s}: {size:6d} bytes ({size/1024:6.2f} KB)")
    
    # Timing arrays (u16 = 2 bytes each)
    print()
    print("TIMING ARRAYS (u16 = 2 bytes each, per level)")
    print("-" * 70)
    timing_per_level = TSC_SIZE_WORDS * 2  # 3 words * 2 bytes = 6 bytes per level
    timing_arrays = {
        "timing_smooth": LEVELS * timing_per_level,
        "timing_apply_op": LEVELS * timing_per_level,
        "timing_residual": LEVELS * timing_per_level,
        "timing_compute": LEVELS * timing_per_level,
        "timing_communication": LEVELS * timing_per_level,
        "timing_restrict": LEVELS * timing_per_level,
        "timing_interp": LEVELS * timing_per_level,
        "timing_setup_init": LEVELS * timing_per_level,
        "timing_rho_check": LEVELS * timing_per_level,
    }
    for name, size in timing_arrays.items():
        print(f"  {name:20s}: {size:6d} bytes ({size/1024:6.2f} KB)")
    
    # Single-level timing buffers
    print()
    print("SINGLE-LEVEL TIMING BUFFERS (u16 = 2 bytes each)")
    print("-" * 70)
    single_timing = {
        "time_total_start_end_u16": 1 * timing_per_level,
        "time_ref_u16": timing_per_level,
        "time_h2d_u16": timing_per_level,
        "time_d2h_u16": timing_per_level,
        "tscStartBuffer": timing_per_level,
        "tscEndBuffer": timing_per_level,
    }
    for name, size in single_timing.items():
        print(f"  {name:20s}: {size:6d} bytes ({size/1024:6.2f} KB)")
    
    # Counter arrays (u16 = 2 bytes each)
    print()
    print("COUNTER ARRAYS (u16 = 2 bytes each)")
    print("-" * 70)
    counter_arrays = {
        "counter_smooth": LEVELS * 2,
        "counter_apply_op": LEVELS * 2,
        "counter_residual": LEVELS * 2,
        "counter_restrict": LEVELS * 2,
        "counter_interp": LEVELS * 2,
        "counter_setup_init": LEVELS * 2,
        "counter_rho_check": LEVELS * 2,
        "counter_compute": LEVELS * 2,
        "counter_communication": LEVELS * 2,
    }
    for name, size in counter_arrays.items():
        print(f"  {name:20s}: {size:6d} bytes ({size/1024:6.2f} KB)")
    
    # Other variables
    print()
    print("OTHER VARIABLES")
    print("-" * 70)
    other_vars = {
        "stencil_coeff": 7 * 4,  # [7]f32
        "rho": 1 * 4,  # [1]f32
        "dot": 1 * 4,  # [1]f32 (dummy for sync)
    }
    for name, size in other_vars.items():
        print(f"  {name:20s}: {size:6d} bytes ({size/1024:6.2f} KB)")
    
    # Stencil module buffers (from stencil_3d_7pts/pe.csl)
    print()
    print("=" * 70)
    print("STENCIL MODULE BUFFERS (f32 = 4 bytes each)")
    print("=" * 70)
    stencil_buffers = {
        "west_buf": BLOCK_SIZE * 4,
        "east_buf": BLOCK_SIZE * 4,
        "south_buf": BLOCK_SIZE * 4,
        "north_buf": BLOCK_SIZE * 4,
    }
    for name, size in stencil_buffers.items():
        print(f"  stencil_mod.{name:15s}: {size:6d} bytes ({size/1024:6.2f} KB)")
    
    # Reduce module buffers
    print()
    print("=" * 70)
    print("REDUCE MODULE BUFFERS")
    print("=" * 70)
    MAX_PENCIL_DIM = 512  # From allreduce/pe.csl
    reduce_buffers = {
        "zero_buf": MAX_PENCIL_DIM * 4,  # f32
        "tscRefBuffer": TSC_SIZE_WORDS * 2,  # u16
    }
    for name, size in reduce_buffers.items():
        print(f"  reduce_mod.{name:15s}: {size:6d} bytes ({size/1024:6.2f} KB)")
    
    # Timer module buffers (from timer_modified.csl)
    print()
    print("=" * 70)
    print("TIMER MODULE BUFFERS")
    print("=" * 70)
    timer_buffers = {
        "tscBufferStart": TIMER_COUNT * TSC_SIZE_WORDS * 2,  # [timerCount, timestamp_size]u16
        "tscBufferStop": TIMER_COUNT * TSC_SIZE_WORDS * 2,
    }
    for name, size in timer_buffers.items():
        print(f"  timer.{name:15s}: {size:6d} bytes ({size/1024:6.2f} KB)")
    
    # Calculate totals
    print()
    print("=" * 70)
    print("TOTAL CALCULATIONS")
    print("=" * 70)
    
    total_main = sum(main_arrays.values())
    total_gmg = sum(gmg_arrays.values())
    total_level = sum(level_arrays.values())
    total_timing = sum(timing_arrays.values())
    total_single_timing = sum(single_timing.values())
    total_counters = sum(counter_arrays.values())
    total_other = sum(other_vars.values())
    total_stencil = sum(stencil_buffers.values())
    total_reduce = sum(reduce_buffers.values())
    total_timer = sum(timer_buffers.values())
    
    total_data = (total_main + total_gmg + total_level + total_timing + 
                  total_single_timing + total_counters + total_other + 
                  total_stencil + total_reduce + total_timer)
    
    print(f"Main arrays:              {total_main:6d} bytes ({total_main/1024:6.2f} KB)")
    print(f"GMG save arrays:          {total_gmg:6d} bytes ({total_gmg/1024:6.2f} KB)")
    print(f"Level parameter arrays:   {total_level:6d} bytes ({total_level/1024:6.2f} KB)")
    print(f"Timing arrays:            {total_timing:6d} bytes ({total_timing/1024:6.2f} KB)")
    print(f"Single timing buffers:    {total_single_timing:6d} bytes ({total_single_timing/1024:6.2f} KB)")
    print(f"Counter arrays:           {total_counters:6d} bytes ({total_counters/1024:6.2f} KB)")
    print(f"Other variables:          {total_other:6d} bytes ({total_other/1024:6.2f} KB)")
    print(f"Stencil module buffers:    {total_stencil:6d} bytes ({total_stencil/1024:6.2f} KB)")
    print(f"Reduce module buffers:    {total_reduce:6d} bytes ({total_reduce/1024:6.2f} KB)")
    print(f"Timer module buffers:     {total_timer:6d} bytes ({total_timer/1024:6.2f} KB)")
    print("-" * 70)
    print(f"TOTAL DATA (calculated):  {total_data:6d} bytes ({total_data/1024:6.2f} KB)")
    print()
    
    # Compare with actual output
    print("=" * 70)
    print("COMPARISON WITH ACTUAL OUTPUT")
    print("=" * 70)
    actual_data = 17072  # From output: "Data (OBJECT symbols): 17072 bytes"
    actual_code = 28008  # From output: "Code (FUNC symbols): 28008 bytes"
    actual_total = 45080  # From output: "Total accounted: 45080 bytes"
    actual_overhead = 312  # From output: "Overhead (stack/other): 312 bytes"
    
    print(f"Actual Data (from ELF):   {actual_data:6d} bytes ({actual_data/1024:6.2f} KB)")
    print(f"Calculated Data:           {total_data:6d} bytes ({total_data/1024:6.2f} KB)")
    print(f"Difference:                {actual_data - total_data:6d} bytes ({(actual_data - total_data)/1024:6.2f} KB)")
    print()
    print(f"Actual Code:               {actual_code:6d} bytes ({actual_code/1024:6.2f} KB)")
    print(f"Actual Total:              {actual_total:6d} bytes ({actual_total/1024:6.2f} KB)")
    print(f"Actual Overhead:           {actual_overhead:6d} bytes ({actual_overhead/1024:6.2f} KB)")
    print()
    
    # Analysis
    print("=" * 70)
    print("ANALYSIS")
    print("=" * 70)
    diff = actual_data - total_data
    if abs(diff) < 100:
        print("✓ Memory usage matches closely! Difference is likely due to:")
        print("  - DSD (Data Structure Descriptor) overhead")
        print("  - Pointer variables")
        print("  - Compiler alignment/padding")
        print("  - Other small variables not accounted for")
    else:
        print(f"⚠ Significant difference: {diff} bytes")
        print("  Possible reasons:")
        print("  - Missing variables in calculation")
        print("  - DSD overhead")
        print("  - Compiler optimizations or padding")
        print("  - Additional module buffers")
    
    print()
    print("=" * 70)
    print("MEMORY OPTIMIZATION OPPORTUNITIES")
    print("=" * 70)
    print("1. REDUCE MODULE - zero_buf:")
    print(f"   Current: {reduce_buffers['zero_buf']} bytes (MAX_PENCIL_DIM=512)")
    print(f"   Could be: {MAX_ZDIM * 4} bytes (MAX_ZDIM={MAX_ZDIM})")
    print(f"   Savings: {reduce_buffers['zero_buf'] - MAX_ZDIM * 4} bytes")
    print()
    print("2. STENCIL MODULE - Communication buffers:")
    print(f"   Current: {total_stencil} bytes (BLOCK_SIZE={BLOCK_SIZE})")
    print(f"   Note: These are needed for communication, but could be optimized")
    print("   if BLOCK_SIZE < MAX_ZDIM in some cases")
    print()
    print("3. TIMING ARRAYS:")
    print(f"   Current: {total_timing} bytes for {len(timing_arrays)} arrays")
    print("   Could potentially be reduced if not all timing is needed")
    print()
    print("4. GMG SAVE ARRAYS:")
    print(f"   Current: {total_gmg} bytes")
    print("   These are necessary for V-cycle state saving")
    print()
    print("5. Consider using smaller data types where possible:")
    print("   - Some counters could be u8 instead of u16 if values < 256")
    print("   - Some timing could be reduced if precision allows")

if __name__ == "__main__":
    analyze_memory()


