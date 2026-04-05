#!/usr/bin/env python3
"""
Wafer Utilization Analysis for GMG V-Cycle on WSE-3.

Computes:
  - Active PE counts per grid size and per multigrid level
  - Wafer utilization as fraction of 893,064 total PEs
  - Memory breakdown (code vs data) from existing ELF analysis
  - What limits scaling to larger problem sizes

Usage:
  python wafer_utilization.py
"""

import os
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(SCRIPT_DIR, '..')

TOTAL_PES = 893_064  # WSE-3 total PEs
PE_SRAM_BYTES = 49_152  # 48 KB per PE
WSE3_DIMS = (762, 1172)  # height x width of usable fabric

GRID_SIZE_NUMS = [4, 8, 16, 32, 64, 128, 256, 512]


def parse_memory_from_responses(filepath):
    """Parse memory usage from all_responses_*.txt.

    Returns dict: grid_label -> {code_bytes, data_bytes, total_bytes, overhead_bytes}
    """
    results = {}
    if not os.path.exists(filepath):
        return results

    with open(filepath, 'r') as f:
        content = f.read()

    blocks = re.split(r'^Output directory: ', content, flags=re.MULTILINE)

    for block in blocks[1:]:
        m = re.match(r'(?:shallow_)?out_dir_S(\d+)x_', block)
        if not m:
            continue
        size = int(m.group(1))
        grid_label = f'{size}x{size}'

        # Extract first memory entry (PE 0,0)
        m_code = re.search(r'Code \(FUNC symbols\):\s+(\d+)\s+bytes', block)
        m_data = re.search(r'Data \(OBJECT symbols\):\s+(\d+)\s+bytes', block)
        m_total = re.search(r'Total accounted:\s+(\d+)\s+bytes', block)
        m_over = re.search(r'Overhead \(stack/other\):\s+(\d+)\s+bytes', block)

        if m_code and m_data:
            results[grid_label] = {
                'code_bytes': int(m_code.group(1)),
                'data_bytes': int(m_data.group(1)),
                'total_bytes': int(m_total.group(1)) if m_total else 0,
                'overhead_bytes': int(m_over.group(1)) if m_over else 0,
            }

    return results


def compute_utilization_table():
    """Compute PE utilization for each grid size and level."""
    print("=" * 120)
    print("WAFER UTILIZATION ANALYSIS")
    print("=" * 120)

    print(f"\nWSE-3 specs: {TOTAL_PES:,} total PEs, {WSE3_DIMS[0]}×{WSE3_DIMS[1]} fabric, "
          f"{PE_SRAM_BYTES:,} bytes ({PE_SRAM_BYTES/1024:.0f} KB) SRAM per PE")

    print(f"\n{'Grid':<12} {'PEs Used':>10} {'% Wafer':>10} {'Levels':>7} "
          f"{'Coarsest PEs':>13} {'Coarsest %':>11} {'Max nx':>7} {'Max ny':>7}")
    print("-" * 90)

    utilization_data = []

    for size in GRID_SIZE_NUMS:
        # Deep V-cycle: levels = log2(size) + 1
        levels_deep = int(np.log2(size)) + 1
        # Shallow V-cycle: levels = log2(size) - 1 (for sizes >= 16)
        levels_shallow = max(2, int(np.log2(size)) - 1) if size >= 16 else levels_deep

        total_pes = size * size
        pct_wafer = 100.0 * total_pes / TOTAL_PES

        # Coarsest level active PEs (deep)
        coarsest_factor = 2 ** (levels_deep - 1)
        coarsest_pes = max(1, (size // coarsest_factor) ** 2)
        coarsest_pct = 100.0 * coarsest_pes / total_pes

        print(f"{size}x{size:<7} {total_pes:>10,} {pct_wafer:>9.4f}% {levels_deep:>7} "
              f"{coarsest_pes:>13,} {coarsest_pct:>10.2f}% {size:>7} {size:>7}")

        utilization_data.append({
            'size': size,
            'total_pes': total_pes,
            'pct_wafer': pct_wafer,
            'levels_deep': levels_deep,
            'coarsest_pes': coarsest_pes,
        })

    # Per-level breakdown for 512x512
    print("\n" + "=" * 90)
    print("PER-LEVEL BREAKDOWN: 512x512 grid (9 levels, deep V-cycle)")
    print("=" * 90)
    print(f"\n{'Level':>6} {'Factor':>7} {'Active PEs':>12} {'% of Total':>11} "
          f"{'nx':>6} {'ny':>6} {'nz':>6} {'Hop Dist':>9}")
    print("-" * 70)

    size = 512
    levels = 9
    for l in range(levels):
        factor = 2 ** l
        active_nx = size // factor
        active_ny = size // factor
        active_nz = size // factor  # z-dim also halves
        active_pes = active_nx * active_ny
        pct = 100.0 * active_pes / (size * size)
        hop = factor  # communication distance in hops

        print(f"{l:>6} {factor:>7} {active_pes:>12,} {pct:>10.4f}% "
              f"{active_nx:>6} {active_ny:>6} {active_nz:>6} {hop:>9}")

    # Shallow variant (7 levels for 512)
    print("\n" + "=" * 90)
    print("PER-LEVEL BREAKDOWN: 512x512 grid (7 levels, shallow V-cycle)")
    print("=" * 90)
    print(f"\n{'Level':>6} {'Factor':>7} {'Active PEs':>12} {'% of Total':>11} "
          f"{'nx':>6} {'ny':>6} {'nz':>6} {'Hop Dist':>9}")
    print("-" * 70)

    levels_shallow = 7
    for l in range(levels_shallow):
        factor = 2 ** l
        active_nx = size // factor
        active_ny = size // factor
        active_nz = size // factor
        active_pes = active_nx * active_ny
        pct = 100.0 * active_pes / (size * size)
        hop = factor

        print(f"{l:>6} {factor:>7} {active_pes:>12,} {pct:>10.4f}% "
              f"{active_nx:>6} {active_ny:>6} {active_nz:>6} {hop:>9}")

    return utilization_data


def print_memory_table():
    """Print memory breakdown from existing data."""
    print("\n" + "=" * 100)
    print("MEMORY BREAKDOWN PER PE (from ELF analysis)")
    print("=" * 100)

    # Parse from 6/6/6 config (representative)
    filepath = os.path.join(BASE_DIR, 'build', 'all_responses_6_6_6.txt')
    mem_data = parse_memory_from_responses(filepath)

    print(f"\n{'Grid':<12} {'Code (KB)':>10} {'Data (KB)':>10} {'Total (KB)':>11} "
          f"{'Overhead':>10} {'Free (KB)':>10} {'Code %':>8} {'Data %':>8} {'Used %':>8}")
    print("-" * 95)

    memory_results = []
    for size in GRID_SIZE_NUMS:
        grid = f'{size}x{size}'
        m = mem_data.get(grid)
        if m:
            code_kb = m['code_bytes'] / 1024
            data_kb = m['data_bytes'] / 1024
            total_kb = m['total_bytes'] / 1024
            overhead_kb = m['overhead_bytes'] / 1024
            free_kb = PE_SRAM_BYTES / 1024 - total_kb - overhead_kb
            code_pct = 100.0 * m['code_bytes'] / PE_SRAM_BYTES
            data_pct = 100.0 * m['data_bytes'] / PE_SRAM_BYTES
            used_pct = 100.0 * (m['total_bytes'] + m['overhead_bytes']) / PE_SRAM_BYTES

            print(f"{grid:<12} {code_kb:>10.2f} {data_kb:>10.2f} {total_kb:>11.2f} "
                  f"{overhead_kb:>10.2f} {free_kb:>10.2f} {code_pct:>7.1f}% {data_pct:>7.1f}% {used_pct:>7.1f}%")

            memory_results.append({
                'size': size, 'code_bytes': m['code_bytes'],
                'data_bytes': m['data_bytes'], 'total_bytes': m['total_bytes'],
            })

    # Scaling limits analysis
    print("\n" + "=" * 80)
    print("SCALING LIMITS ANALYSIS")
    print("=" * 80)
    print(f"""
What limits scaling beyond 512×512:

1. CODE SIZE: The CSL compiler aggressively inlines functions for the state machine
   kernel. Code occupies ~22-23 KB ({100*23000/PE_SRAM_BYTES:.0f}% of 48 KB) and is
   nearly CONSTANT across problem sizes (dominated by stencil + allreduce libraries).

2. DATA SIZE: Grows with problem size and multigrid levels. At 512×512 (9 levels),
   per-PE data includes:
   - Solution vectors (u, f, r, Au): 4 × MAX_ZDIM floats
   - GMG save arrays: ~2 × MAX_ZDIM floats (geometric series)
   - Timing arrays: ~13 × LEVELS × 3 words
   - Level parameters: 4 × LEVELS floats

3. GEOMETRIC SERIES: Total data per array = nz + nz/2 + nz/4 + ... = nz × (2 - 1/2^(L-1))
   For nz=512, L=9: ~1023 floats ≈ 4 KB per save array

4. PRACTICAL LIMIT: At 512×512, total used ~{memory_results[-1]['total_bytes']/1024:.1f} KB
   of 48 KB. Beyond 512, the z-dimension and level count would push data past available SRAM.

5. FABRIC CONSTRAINT: 512×512 = 262,144 PEs uses {100*262144/TOTAL_PES:.1f}% of the wafer.
   The usable fabric is {WSE3_DIMS[0]}×{WSE3_DIMS[1]} = {WSE3_DIMS[0]*WSE3_DIMS[1]:,} PEs.
   Theoretically up to ~762×762 = 580,644 PEs could be used if memory permits.
""")

    return memory_results


def plot_utilization(utilization_data, memory_results, output_dir):
    """Generate utilization figure."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    sizes = [d['size'] for d in utilization_data]
    total_pes = [d['total_pes'] for d in utilization_data]
    pct_wafer = [d['pct_wafer'] for d in utilization_data]

    # --- Left: PE count ---
    ax = axes[0]
    ax.bar(range(len(sizes)), total_pes, color='#2196F3', edgecolor='black', linewidth=0.5)
    ax.axhline(y=TOTAL_PES, color='red', linestyle='--', linewidth=1.5, label=f'Total WSE-3 ({TOTAL_PES:,})')
    ax.set_xticks(range(len(sizes)))
    ax.set_xticklabels([f'{s}x{s}' for s in sizes], fontsize=10, rotation=45)
    ax.set_ylabel('Active PEs (fine level)', fontsize=12)
    ax.set_title('PE Utilization', fontsize=14)
    ax.set_yscale('log')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    # --- Center: Wafer % ---
    ax2 = axes[1]
    bars = ax2.bar(range(len(sizes)), pct_wafer, color='#4CAF50', edgecolor='black', linewidth=0.5)
    for bar, pct in zip(bars, pct_wafer):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f'{pct:.2f}%', ha='center', fontsize=9, fontweight='bold')
    ax2.set_xticks(range(len(sizes)))
    ax2.set_xticklabels([f'{s}x{s}' for s in sizes], fontsize=10, rotation=45)
    ax2.set_ylabel('% of Wafer Used', fontsize=12)
    ax2.set_title('Wafer Utilization %', fontsize=14)
    ax2.set_ylim(0, max(pct_wafer) * 1.3)
    ax2.grid(axis='y', alpha=0.3)

    # --- Right: Memory stacked bar ---
    ax3 = axes[2]
    if memory_results:
        mem_sizes = [d['size'] for d in memory_results]
        code_kb = [d['code_bytes'] / 1024 for d in memory_results]
        data_kb = [d['data_bytes'] / 1024 for d in memory_results]
        free_kb = [PE_SRAM_BYTES / 1024 - c - d for c, d in zip(code_kb, data_kb)]

        x = range(len(mem_sizes))
        ax3.bar(x, code_kb, color='#F44336', label='Code', edgecolor='black', linewidth=0.5)
        ax3.bar(x, data_kb, bottom=code_kb, color='#FF9800', label='Data', edgecolor='black', linewidth=0.5)
        ax3.bar(x, free_kb, bottom=[c + d for c, d in zip(code_kb, data_kb)],
                color='#E0E0E0', label='Free', edgecolor='black', linewidth=0.5)
        ax3.axhline(y=48, color='black', linestyle='-', linewidth=2, label='48 KB limit')
        ax3.set_xticks(x)
        ax3.set_xticklabels([f'{s}x{s}' for s in mem_sizes], fontsize=10, rotation=45)
        ax3.set_ylabel('Memory (KB)', fontsize=12)
        ax3.set_title('Per-PE Memory Breakdown', fontsize=14)
        ax3.legend(fontsize=9)
        ax3.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    outpath = os.path.join(output_dir, 'wafer_utilization.png')
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {outpath}")


def main():
    utilization_data = compute_utilization_table()
    memory_results = print_memory_table()
    plot_utilization(utilization_data, memory_results, SCRIPT_DIR)
    print("\nDone!")


if __name__ == '__main__':
    main()
