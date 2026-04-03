#!/usr/bin/env python3
"""
Roofline FLOP Analysis for GMG V-Cycle on WSE-3.

Parses roofline counter data from response.txt files (after recompilation
with FLOP counters enabled in kernel_gmg_vcycle.csl).

Follows Ruichisai et al. (Table V) methodology for per-operation FLOP
counting on Cerebras WSE.

Usage:
  # After recompiling with counters and running:
  python roofline_analysis.py <response.txt or all_responses.txt>

FLOPs per operation type:
  FSUB: 1 FLOP,  2 loads, 1 store
  FMAC: 2 FLOPs, 3 loads, 1 store  (fused multiply-add)
  FMUL: 1 FLOP,  2 loads, 1 store
  FADD: 1 FLOP,  2 loads, 1 store
  FNEG: 1 FLOP,  1 load,  1 store
  FMOV: 0 FLOPs, 0-1 loads, 1 store
  FMAX: 1 FLOP,  1 load,  1 store
"""

import argparse
import re
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# FLOP weight per operation type (matching ruichisai Table V)
FLOP_PER_OP = {
    'fsub': 1,
    'fmac': 2,   # fused multiply-add = 2 FLOPs
    'fmul': 1,
    'fadd': 1,
    'fneg': 1,
    'fmov': 0,   # data movement, no compute
    'fmax': 1,
}

# Memory traffic per operation (loads + stores) in units of f32 words
MEM_TRAFFIC = {
    'fsub': {'loads': 2, 'stores': 1},
    'fmac': {'loads': 3, 'stores': 1},
    'fmul': {'loads': 2, 'stores': 1},
    'fadd': {'loads': 2, 'stores': 1},
    'fneg': {'loads': 1, 'stores': 1},
    'fmov': {'loads': 1, 'stores': 1},
    'fmax': {'loads': 1, 'stores': 1},
}

# WSE-3 machine parameters (from Ryuichi / Sai et al.)
# Per-PE: 875 MHz clock, ~0.850 GFLOPS/PE
# Full wafer: 893,064 PEs × 0.850 GFLOPS = 759.1 TFLOPS peak
# Memory BW: 21 PB/s (full wafer), ~23.5 KB/cycle/PE
PE_PEAK_GFLOPS = 0.850    # GFLOPS per PE at 875 MHz
WSE3_FULL_WAFER_PES = 893_064
WSE3_PEAK_FLOPS = WSE3_FULL_WAFER_PES * PE_PEAK_GFLOPS * 1e9  # ~759 TFLOPS
WSE3_MEM_BW = 21e15       # 21 PB/s memory bandwidth (full wafer)


def parse_device_counters(filepath):
    """Parse FLOP counter data from response.txt / all_responses.txt.

    Looks for the roofline table output from run_gmg_vcycle.py:
      Level N:
      FSUB      count  flop/op  total_flops  mem_traffic
      ...

    Returns dict: grid_label -> {level -> {op -> count}}
    """
    results = {}
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return results

    with open(filepath, 'r') as f:
        content = f.read()

    blocks = re.split(r'^Output directory: ', content, flags=re.MULTILINE)

    for block in blocks[1:] if len(blocks) > 1 else [content]:
        # Get grid size
        m = re.match(r'(?:shallow_)?out_dir_S(\d+)x_', block)
        if m:
            size = int(m.group(1))
            grid_label = f'{size}x{size}'
        else:
            grid_label = 'unknown'

        # Parse roofline table
        level_counts = {}
        level_pattern = re.compile(
            r'^Level (\d+):\s*\n'
            r'Operation.*\n-+\n'
            r'((?:(?:FSUB|FMAC|FMUL|FADD|FNEG|FMOV|FMAX)\s+[\d,]+.*\n)*)',
            re.MULTILINE
        )

        for match in level_pattern.finditer(block):
            level = int(match.group(1))
            ops_text = match.group(2)
            ops = {}
            for line in ops_text.strip().split('\n'):
                parts = line.split()
                if len(parts) >= 2:
                    op_name = parts[0].lower()
                    count = int(parts[1].replace(',', ''))
                    ops[op_name] = count
            level_counts[level] = ops

        # Parse grand total
        m_flops = re.search(r'Total FLOPs per PE:\s+([\d,]+)', block)
        m_time = re.search(r'V-cycle time \(all iters\):\s+([\d.]+)\s+us', block)
        m_pes = re.search(r'Active PEs \(fine level\):\s+([\d,]+)', block)
        m_achieved = re.search(r'Achieved performance:\s+([\d.e+]+)\s+FLOP/s', block)

        if level_counts:
            results[grid_label] = {
                'levels': level_counts,
                'total_flops_pe': int(m_flops.group(1).replace(',', '')) if m_flops else None,
                'total_time_us': float(m_time.group(1)) if m_time else None,
                'active_pes': int(m_pes.group(1).replace(',', '')) if m_pes else None,
                'achieved_flops': float(m_achieved.group(1)) if m_achieved else None,
            }

    return results


def print_table_v(counts, total_time_us=None, num_active_pes=None):
    """Print Table V style output (like ruichisai paper).

    counts: dict of {level -> {op_type -> count}}
    """
    all_ops = ['fsub', 'fmac', 'fmul', 'fadd', 'fneg', 'fmov', 'fmax']

    print("\n" + "=" * 120)
    print("TABLE V: FLOP Counts per Operation Type per Level (per PE)")
    print("=" * 120)

    for level in sorted(counts.keys()):
        c = counts[level]
        print(f"\nLevel {level}:")
        print(f"{'Operation':<10} {'Count':>15} {'FLOP/op':>8} {'Total FLOPs':>15} "
              f"{'Loads':>8} {'Stores':>8} {'Mem Traffic (B)':>16}")
        print("-" * 85)

        level_total_flops = 0
        level_total_mem = 0

        for op in all_ops:
            count = c.get(op, 0)
            flop = FLOP_PER_OP[op]
            total_flops = count * flop
            loads = MEM_TRAFFIC[op]['loads'] * count
            stores = MEM_TRAFFIC[op]['stores'] * count
            mem_bytes = (loads + stores) * 4  # f32 = 4 bytes

            level_total_flops += total_flops
            level_total_mem += mem_bytes

            if count > 0:
                print(f"{op.upper():<10} {count:>15,} {flop:>8} {total_flops:>15,} "
                      f"{loads:>8,} {stores:>8,} {mem_bytes:>16,}")

        ai = level_total_flops / level_total_mem if level_total_mem > 0 else 0
        print("-" * 85)
        print(f"{'TOTAL':<10} {'':>15} {'':>8} {level_total_flops:>15,} "
              f"{'':>8} {'':>8} {level_total_mem:>16,}")
        print(f"Memory Arithmetic Intensity: {ai:.4f} FLOP/Byte")

    # Grand total
    print("\n" + "=" * 120)
    print("GRAND TOTAL (all levels)")
    print("=" * 120)

    grand_counts = {op: 0 for op in all_ops}
    for level_c in counts.values():
        for op in all_ops:
            grand_counts[op] += level_c.get(op, 0)

    grand_total_flops = 0
    grand_total_mem = 0

    print(f"{'Operation':<10} {'Count':>15} {'FLOP/op':>8} {'Total FLOPs':>15} {'Mem Traffic (B)':>16}")
    print("-" * 70)

    for op in all_ops:
        count = grand_counts[op]
        flop = FLOP_PER_OP[op]
        total_flops = count * flop
        mem_bytes = (MEM_TRAFFIC[op]['loads'] + MEM_TRAFFIC[op]['stores']) * count * 4
        grand_total_flops += total_flops
        grand_total_mem += mem_bytes

        if count > 0:
            print(f"{op.upper():<10} {count:>15,} {flop:>8} {total_flops:>15,} {mem_bytes:>16,}")

    grand_ai = grand_total_flops / grand_total_mem if grand_total_mem > 0 else 0
    print("-" * 70)
    print(f"{'TOTAL':<10} {'':>15} {'':>8} {grand_total_flops:>15,} {grand_total_mem:>16,}")
    print(f"\nTotal FLOPs per PE:             {grand_total_flops:,}")
    print(f"Total Memory Traffic per PE:    {grand_total_mem:,} bytes")
    print(f"Memory Arithmetic Intensity:    {grand_ai:.4f} FLOP/Byte")

    if total_time_us:
        total_time_s = total_time_us * 1e-6
        pe_achieved = grand_total_flops / total_time_s
        print(f"\n--- PE(0,0) Performance ---")
        print(f"V-cycle time (all iters):       {total_time_us:.3f} us")
        print(f"PE(0,0) achieved:               {pe_achieved:.3e} FLOP/s ({pe_achieved/1e9:.3f} GFLOP/s)")
        print(f"Per-PE peak:                    {PE_PEAK_GFLOPS} GFLOP/s (875 MHz)")
        print(f"% of per-PE peak:               {100 * pe_achieved / (PE_PEAK_GFLOPS * 1e9):.2f}%")


def compute_summary(all_results):
    """Compute per-grid summary using PE(0,0) counters only (no PE scaling).

    PE(0,0) is active at every level. Its per-level counters represent
    the work done by one active PE at that level.
    """
    summary = {}
    all_ops = ['fsub', 'fmac', 'fmul', 'fadd', 'fneg', 'fmov', 'fmax']

    for grid_label, data in all_results.items():
        size = int(grid_label.split('x')[0])
        num_levels = len(data['levels'])
        time_us = data.get('total_time_us', 0)
        time_s = time_us * 1e-6 if time_us else 0

        pe00_total_flops = 0
        pe00_total_mem = 0
        level_details = []

        for level in sorted(data['levels'].keys()):
            level_c = data['levels'][level]
            active_pes = (size // (2 ** level)) ** 2
            nz = size >> level

            level_flops = 0
            level_mem = 0
            for op in all_ops:
                count = level_c.get(op, 0)
                level_flops += count * FLOP_PER_OP[op]
                level_mem += count * (MEM_TRAFFIC[op]['loads'] + MEM_TRAFFIC[op]['stores']) * 4

            pe00_total_flops += level_flops
            pe00_total_mem += level_mem

            level_ai = level_flops / level_mem if level_mem > 0 else 0
            level_details.append({
                'level': level, 'nz': nz, 'active_pes': active_pes,
                'flops': level_flops, 'mem': level_mem, 'ai': level_ai,
            })

        pe00_ai = pe00_total_flops / pe00_total_mem if pe00_total_mem > 0 else 0
        pe00_achieved = pe00_total_flops / time_s if time_s > 0 else 0

        summary[grid_label] = {
            'size': size,
            'num_levels': num_levels,
            'pe00_flops': pe00_total_flops,
            'pe00_mem': pe00_total_mem,
            'pe00_ai': pe00_ai,
            'time_us': time_us,
            'pe00_achieved': pe00_achieved,
            'pct_pe_peak': 100 * pe00_achieved / (PE_PEAK_GFLOPS * 1e9) if pe00_achieved > 0 else 0,
            'level_details': level_details,
        }
    return summary


def plot_roofline(summary, output_dir):
    """Plot per-PE roofline for PE(0,0) following Sai et al. methodology.

    Left: PE(0,0) roofline across grid sizes (one point per grid).
    Right: Per-level breakdown for the largest grid.

    Ceiling: per-PE peak (0.850 GFLOP/s) + per-PE memory BW slope.
    """
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    sorted_grids = sorted(summary.keys(), key=lambda g: int(g.split('x')[0]))
    largest_grid = sorted_grids[-1] if sorted_grids else None

    # Per-PE memory BW: 21 PB/s / 893,064 PEs ≈ 23.5 MB/PE/cycle...
    # Actually: 21e15 bytes/s / 893,064 PEs = 23.5 GB/s per PE
    PE_MEM_BW = WSE3_MEM_BW / WSE3_FULL_WAFER_PES  # bytes/s per PE
    PE_PEAK = PE_PEAK_GFLOPS * 1e9  # FLOP/s per PE

    ai_range = np.logspace(-3, 2, 500)
    mem_ceiling = PE_MEM_BW * ai_range
    compute_ceiling = np.full_like(ai_range, PE_PEAK)
    roofline = np.minimum(mem_ceiling, compute_ceiling)

    # --- Left: PE(0,0) roofline across grid sizes ---
    ax = axes[0]

    ax.loglog(ai_range, roofline, 'k-', linewidth=2.5, label='Per-PE Roofline')
    ax.loglog(ai_range, mem_ceiling, 'b-', linewidth=1, alpha=0.3)
    ax.axhline(y=PE_PEAK, color='r', linestyle='--', linewidth=1.5, alpha=0.5,
               label=f'PE peak ({PE_PEAK_GFLOPS} GFLOP/s)')

    colors_grid = plt.cm.viridis(np.linspace(0.2, 0.9, len(sorted_grids)))
    for i, g in enumerate(sorted_grids):
        s = summary[g]
        if s['pe00_achieved'] > 0:
            ax.plot(s['pe00_ai'], s['pe00_achieved'], 'o', color=colors_grid[i],
                    markersize=12, markeredgecolor='black', markeredgewidth=1.5, zorder=5)
            ax.annotate(f"{g}\n({s['pct_pe_peak']:.1f}%)",
                        (s['pe00_ai'], s['pe00_achieved']),
                        textcoords="offset points", xytext=(10, 5), fontsize=9,
                        color=colors_grid[i], fontweight='bold')

    ax.set_xlabel('Arithmetic Intensity (FLOP/Byte)', fontsize=14)
    ax.set_ylabel('PE(0,0) Performance (FLOP/s)', fontsize=14)
    ax.set_title('Per-PE Roofline: GMG V-Cycle on WSE-3', fontsize=15)
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(alpha=0.3, which='both')
    ax.set_xlim(0.01, 100)
    ax.set_ylim(1e4, PE_PEAK * 10)
    ax.tick_params(axis='both', labelsize=11)

    # --- Right: Per-level breakdown for largest grid ---
    ax2 = axes[1]

    ax2.loglog(ai_range, roofline, 'k-', linewidth=2.5, label='Per-PE Roofline')
    ax2.loglog(ai_range, mem_ceiling, 'b-', linewidth=1, alpha=0.3)
    ax2.axhline(y=PE_PEAK, color='r', linestyle='--', linewidth=1.5, alpha=0.5,
                label=f'PE peak ({PE_PEAK_GFLOPS} GFLOP/s)')

    if largest_grid:
        s = summary[largest_grid]
        levels = s['level_details']
        time_s = s['time_us'] * 1e-6 if s['time_us'] else 1

        colors_level = plt.cm.plasma(np.linspace(0.1, 0.9, len(levels)))
        for ld, color in zip(levels, colors_level):
            # Per-PE achieved at this level = level_flops / total_time
            # (PE(0,0) does work at all levels; total_time covers all of them)
            level_pe_perf = ld['flops'] / time_s if time_s > 0 else 0
            if level_pe_perf > 0 and ld['ai'] > 0:
                ax2.plot(ld['ai'], level_pe_perf, 's', color=color, markersize=12,
                         markeredgecolor='black', markeredgewidth=1.5, zorder=5)
                ax2.annotate(f"L{ld['level']} (nz={ld['nz']})",
                             (ld['ai'], level_pe_perf),
                             textcoords="offset points", xytext=(8, 5), fontsize=9,
                             color=color, fontweight='bold')

        # Total PE(0,0) point
        if s['pe00_achieved'] > 0:
            ax2.plot(s['pe00_ai'], s['pe00_achieved'], '*', color='red',
                     markersize=18, markeredgecolor='black', markeredgewidth=1.5, zorder=6)
            ax2.annotate(f"Total ({s['pct_pe_peak']:.1f}%)",
                         (s['pe00_ai'], s['pe00_achieved']),
                         textcoords="offset points", xytext=(10, -15), fontsize=10,
                         color='red', fontweight='bold')

        ax2.set_title(f'Per-Level: {largest_grid} ({s["num_levels"]} levels)', fontsize=15)

    ax2.set_xlabel('Arithmetic Intensity (FLOP/Byte)', fontsize=14)
    ax2.set_ylabel('PE(0,0) Performance (FLOP/s)', fontsize=14)
    ax2.legend(fontsize=10, loc='upper left')
    ax2.grid(alpha=0.3, which='both')
    ax2.set_xlim(0.01, 100)
    ax2.set_ylim(1e4, PE_PEAK * 10)
    ax2.tick_params(axis='both', labelsize=11)

    plt.tight_layout()
    outpath = os.path.join(output_dir, 'roofline_plot.png')
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {outpath}")


def print_summary_table(summary):
    """Print PE(0,0) roofline summary across all grid sizes."""
    print("\n" + "=" * 120)
    print("PE(0,0) ROOFLINE SUMMARY ACROSS GRID SIZES")
    print(f"Per-PE peak: {PE_PEAK_GFLOPS} GFLOP/s at 875 MHz")
    print("=" * 120)
    print(f"{'Grid':<10} {'Lvls':>5} {'FLOPs':>12} {'Mem(B)':>12} {'AI':>10} "
          f"{'Time(us)':>12} {'GFLOP/s':>10} {'% PE peak':>10}")
    print("-" * 85)

    sorted_grids = sorted(summary.keys(), key=lambda g: int(g.split('x')[0]))
    for g in sorted_grids:
        s = summary[g]
        print(f"{g:<10} {s['num_levels']:>5} {s['pe00_flops']:>12,} {s['pe00_mem']:>12,} "
              f"{s['pe00_ai']:>10.4f} {s['time_us']:>12.3f} "
              f"{s['pe00_achieved']/1e9:>10.3f} {s['pct_pe_peak']:>9.2f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', nargs='?', help='response.txt or all_responses.txt with device counter data')
    args = parser.parse_args()

    print("=" * 120)
    print("ROOFLINE FLOP ANALYSIS: GMG V-Cycle on WSE-3")
    print("Following Ruichisai et al. Table V methodology")
    print("=" * 120)

    if args.input:
        all_results = parse_device_counters(args.input)
        if all_results:
            for grid_label, data in sorted(all_results.items(),
                                            key=lambda x: int(x[0].split('x')[0])):
                print(f"\n{'#' * 120}")
                print(f"# Grid: {grid_label}")
                print(f"{'#' * 120}")
                print_table_v(
                    data['levels'],
                    total_time_us=data.get('total_time_us'),
                    num_active_pes=data.get('active_pes'),
                )

            summary = compute_summary(all_results)
            print_summary_table(summary)
            plot_roofline(summary, SCRIPT_DIR)
        else:
            print("No roofline counter data found in the file.")
            print("Make sure the file contains output from a run with FLOP counters enabled.")
    else:
        print("\nUsage: python roofline_analysis.py <response.txt or all_responses.txt>")
        print("\nPrerequisites:")
        print("  1. Recompile (kernel_gmg_vcycle.csl already instrumented with FLOP counters)")
        print("  2. Run on device: python compile_and_run_wse3.py --only-device")
        print("  3. Pass response.txt to this script")


if __name__ == '__main__':
    main()
