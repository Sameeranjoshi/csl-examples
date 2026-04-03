#!/usr/bin/env python3
"""
Time-to-Solution (TTS) Analysis for GMG V-Cycle on WSE-3 vs GH200.

Extracts iteration counts and per-V-cycle times from:
  - all_responses_*.txt (WSE-3 data)
  - h200_vs_cs3_feb7.csv (GH200 baseline data)

Produces:
  1. TTS comparison table (printed)
  2. TTS bar chart figure (time_to_solution.png)
  3. TTS speedup figure (tts_speedup.png)

Usage:
  python time_to_solution.py
"""

import os
import re
import csv
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(SCRIPT_DIR, '..')

# Grid sizes in order
GRID_SIZES = ['4x4', '8x8', '16x16', '32x32', '64x64', '128x128', '256x256', '512x512']
GRID_SIZE_NUMS = [4, 8, 16, 32, 64, 128, 256, 512]

# WSE-3 config files mapping: (label, filename)
WSE3_CONFIGS = [
    ('6/6/100', 'all_responses_6_6_100.txt'),
    ('4/4/6',   'all_responses_4_4_6.txt'),
    ('6/6/6',   'all_responses_6_6_6.txt'),
    ('4/4/100', 'all_responses_4_4_100.txt'),
    ('6/6/6(Shallow)', 'all_responses_6_6_6_shallow.txt'),
]


def parse_wse3_responses(filepath):
    """Parse an all_responses_*.txt file to extract per-grid-size TTS data.

    Returns dict: grid_label -> {
        'iterations': int,
        'avg_vcycle_us': float,
        'tts_us': float,
        'tts_s': float,
        'final_rho': float,
        'converged': bool,
        'rho_history': list of floats,
    }
    """
    results = {}
    if not os.path.exists(filepath):
        return results

    with open(filepath, 'r') as f:
        content = f.read()

    # Split into per-output-directory blocks
    blocks = re.split(r'^Output directory: ', content, flags=re.MULTILINE)

    for block in blocks[1:]:  # skip preamble before first block
        # Extract grid size from directory name
        m = re.match(r'out_dir_S(\d+)x_', block) or re.match(r'shallow_out_dir_S(\d+)x_', block)
        if not m:
            continue
        size = int(m.group(1))
        grid_label = f'{size}x{size}'

        # Extract device iterations
        m_iter = re.search(r'Device iterations\s*:\s*(\d+)', block)
        if not m_iter:
            continue
        iterations = int(m_iter.group(1))

        # Extract average V-cycle time in microseconds
        m_avg = re.search(r'1-V cycle time\(Average\)\s*\(us\[cycles\]\)\s*:\s*([\d.]+)us', block)
        if not m_avg:
            continue
        avg_vcycle_us = float(m_avg.group(1))

        # Extract final rho
        m_rho = re.search(r'Device final \|rho\|_inf\s*:\s*([\d.eE+\-]+)', block)
        final_rho = float(m_rho.group(1)) if m_rho else None

        # Extract convergence status
        m_conv = re.search(r'Converged:\s*(Yes|No)', block)
        converged = m_conv.group(1) == 'Yes' if m_conv else None

        # Extract rho history
        rho_history = []
        rho_section = re.search(
            r'Rho values after each iteration.*?Total iterations performed:\s*\d+\s*\n'
            r'Iteration\s+\|rho\|_max\s*\n-+\n(.*?)(?:\[GMG\]|\n\n)',
            block, re.DOTALL
        )
        if rho_section:
            for line in rho_section.group(1).strip().split('\n'):
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        rho_history.append(float(parts[1]))
                    except ValueError:
                        pass

        tts_us = iterations * avg_vcycle_us
        tts_s = tts_us * 1e-6

        results[grid_label] = {
            'iterations': iterations,
            'avg_vcycle_us': avg_vcycle_us,
            'avg_vcycle_s': avg_vcycle_us * 1e-6,
            'tts_us': tts_us,
            'tts_s': tts_s,
            'final_rho': final_rho,
            'converged': converged,
            'rho_history': rho_history,
        }

    return results


def parse_h200_csv(filepath):
    """Parse h200_vs_cs3_feb7.csv to extract GH200 data.

    Returns dict: config_label -> {grid_label -> {iterations, tts_s, per_vcycle_s}}
    """
    results = {}
    if not os.path.exists(filepath):
        return results

    with open(filepath, 'r') as f:
        lines = f.readlines()

    # Row 12 (0-indexed) is the header with column meanings
    # Rows 13-20 are data rows (4x4 through 512x512)
    # H200 configs at columns: (1,2,3)=6/6/100, (4,5,6)=4/4/6, (7,8,9)=6/6/6, (10,11,12)=4/4/100
    h200_configs = {
        '6/6/100': (1, 2, 3),   # FP32(total), ITERATIONS, 1 V cycle
        '4/4/6':   (4, 5, 6),
        '6/6/6':   (7, 8, 9),
        '4/4/100': (10, 11, 12),
    }

    data_rows = []
    for line in lines[12:20]:  # rows for 4x4 through 512x512
        # Parse CSV, handling empty fields
        reader = csv.reader([line])
        row = next(reader)
        data_rows.append(row)

    grid_labels = ['4x4', '8x8', '16x16', '32x32', '64x64', '128x128', '256x256', '512x512']

    for config_label, (col_tts, col_iter, col_vcycle) in h200_configs.items():
        config_data = {}
        for i, grid_label in enumerate(grid_labels):
            if i >= len(data_rows):
                break
            row = data_rows[i]
            try:
                tts_s = float(row[col_tts]) if row[col_tts].strip() else None
                iterations = int(float(row[col_iter])) if row[col_iter].strip() else None
                per_vcycle_s = float(row[col_vcycle]) if row[col_vcycle].strip() else None
            except (ValueError, IndexError):
                tts_s = iterations = per_vcycle_s = None

            if tts_s is not None:
                config_data[grid_label] = {
                    'tts_s': tts_s,
                    'iterations': iterations,
                    'per_vcycle_s': per_vcycle_s,
                }
        results[config_label] = config_data

    return results


def print_tts_table(wse3_all, h200_all):
    """Print comprehensive TTS comparison table."""
    print("=" * 130)
    print("TIME-TO-SOLUTION COMPARISON: WSE-3 (GLOW) vs GH200 (HPGMG)")
    print("=" * 130)

    for config_label, filename in WSE3_CONFIGS:
        wse3 = wse3_all.get(config_label, {})
        h200_config = config_label.replace('(Shallow)', '').strip()
        # Shallow has no H200 counterpart; use 6/6/6 as reference
        if 'Shallow' in config_label:
            h200_config = '6/6/6'
        h200 = h200_all.get(h200_config, {})

        print(f"\n--- Config: {config_label} (pre/post/bottom) ---")
        print(f"{'Grid':<12} {'WSE3 Iters':>10} {'WSE3 TTS(s)':>14} {'WSE3 /Vcyc(us)':>16} "
              f"{'H200 Iters':>10} {'H200 TTS(s)':>14} {'H200 /Vcyc(us)':>16} "
              f"{'TTS Speedup':>12} {'Vcyc Speedup':>13} {'WSE3 Conv':>10}")
        print("-" * 130)

        for grid in GRID_SIZES:
            w = wse3.get(grid, {})
            h = h200.get(grid, {})

            w_iters = w.get('iterations', '-')
            w_tts = f"{w['tts_s']:.6f}" if 'tts_s' in w else '-'
            w_vcyc = f"{w['avg_vcycle_us']:.3f}" if 'avg_vcycle_us' in w else '-'
            w_conv = 'Yes' if w.get('converged') else ('No' if w.get('converged') is False else '-')

            h_iters = h.get('iterations', '-')
            h_tts = f"{h['tts_s']:.6f}" if 'tts_s' in h else '-'
            h_vcyc = f"{h['per_vcycle_s'] * 1e6:.3f}" if h.get('per_vcycle_s') else '-'

            if 'tts_s' in w and 'tts_s' in h and w['tts_s'] > 0:
                tts_speedup = f"{h['tts_s'] / w['tts_s']:.2f}x"
            else:
                tts_speedup = '-'

            if 'avg_vcycle_us' in w and h.get('per_vcycle_s'):
                vcyc_speedup = f"{(h['per_vcycle_s'] * 1e6) / w['avg_vcycle_us']:.2f}x"
            else:
                vcyc_speedup = '-'

            print(f"{grid:<12} {str(w_iters):>10} {w_tts:>14} {w_vcyc:>16} "
                  f"{str(h_iters):>10} {h_tts:>14} {h_vcyc:>16} "
                  f"{tts_speedup:>12} {vcyc_speedup:>13} {w_conv:>10}")


def plot_tts_comparison(wse3_all, h200_all, output_dir):
    """Generate TTS bar chart comparing WSE-3 and GH200."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # --- Left plot: TTS absolute times ---
    ax = axes[0]
    configs_to_plot = ['6/6/6', '6/6/6(Shallow)', '6/6/100']
    colors_wse3 = {'6/6/6': '#2196F3', '6/6/6(Shallow)': '#4CAF50', '6/6/100': '#FF9800'}
    colors_h200 = {'6/6/6': '#90CAF9', '6/6/6(Shallow)': '#A5D6A7', '6/6/100': '#FFCC80'}

    # Only plot sizes with both H200 and WSE3 data
    plot_sizes = ['16x16', '32x32', '64x64', '128x128', '256x256', '512x512']
    x = np.arange(len(plot_sizes))
    bar_width = 0.12
    offset = 0

    for config in configs_to_plot:
        wse3 = wse3_all.get(config, {})
        h200_config = '6/6/6' if 'Shallow' in config else config
        h200 = h200_all.get(h200_config, {})

        wse3_tts = [wse3.get(g, {}).get('tts_s', 0) for g in plot_sizes]
        h200_tts = [h200.get(g, {}).get('tts_s', 0) for g in plot_sizes]

        label_w = f'WSE-3 {config}'
        label_h = f'GH200 {h200_config}'
        ax.bar(x + offset * bar_width, h200_tts, bar_width, label=label_h,
               color=colors_h200[config], edgecolor='black', linewidth=0.5)
        ax.bar(x + (offset + 1) * bar_width, wse3_tts, bar_width, label=label_w,
               color=colors_wse3[config], edgecolor='black', linewidth=0.5)
        offset += 2

    ax.set_xlabel('Problem Size (nx × ny)', fontsize=14)
    ax.set_ylabel('Time-to-Solution (seconds)', fontsize=14)
    ax.set_title('Time-to-Solution: GH200 vs WSE-3', fontsize=16)
    ax.set_xticks(x + 2.5 * bar_width)
    ax.set_xticklabels(plot_sizes, fontsize=12)
    ax.set_yscale('log')
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    ax.tick_params(axis='y', labelsize=12)

    # --- Right plot: TTS Speedup ---
    ax2 = axes[1]
    all_configs = ['6/6/100', '4/4/6', '6/6/6', '4/4/100', '6/6/6(Shallow)']
    config_colors = ['#F44336', '#FF9800', '#2196F3', '#9C27B0', '#4CAF50']

    for config, color in zip(all_configs, config_colors):
        wse3 = wse3_all.get(config, {})
        h200_config = '6/6/6' if 'Shallow' in config else config
        h200 = h200_all.get(h200_config, {})

        speedups = []
        sizes_with_data = []
        for g in plot_sizes:
            w = wse3.get(g, {})
            h = h200.get(g, {})
            if w.get('tts_s') and h.get('tts_s') and w['tts_s'] > 0:
                speedups.append(h['tts_s'] / w['tts_s'])
                sizes_with_data.append(g)

        if speedups:
            ax2.plot(sizes_with_data, speedups, 'o-', label=config, color=color,
                     linewidth=2, markersize=8)
            # Annotate last point
            ax2.annotate(f'{speedups[-1]:.1f}x', (sizes_with_data[-1], speedups[-1]),
                         textcoords="offset points", xytext=(10, 5), fontsize=10,
                         color=color, fontweight='bold')

    ax2.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, label='Baseline (1x)')
    ax2.set_xlabel('Problem Size (nx × ny)', fontsize=14)
    ax2.set_ylabel('TTS Speedup (GH200 / WSE-3)', fontsize=16)
    ax2.set_title('Time-to-Solution Speedup', fontsize=16)
    ax2.legend(fontsize=10, loc='upper left')
    ax2.grid(alpha=0.3)
    ax2.tick_params(axis='both', labelsize=12)

    plt.tight_layout()
    outpath = os.path.join(output_dir, 'time_to_solution.png')
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {outpath}")


def plot_iterations_comparison(wse3_all, h200_all, output_dir):
    """Plot iteration counts side by side to show WSE-3 converges in fewer iterations."""
    fig, ax = plt.subplots(figsize=(12, 6))

    plot_sizes = ['16x16', '32x32', '64x64', '128x128', '256x256', '512x512']
    x = np.arange(len(plot_sizes))
    bar_width = 0.15

    configs = ['6/6/6', '6/6/100', '6/6/6(Shallow)']
    colors_w = ['#2196F3', '#FF9800', '#4CAF50']
    colors_h = ['#90CAF9', '#FFCC80', '#A5D6A7']

    offset = 0
    for config, cw, ch in zip(configs, colors_w, colors_h):
        wse3 = wse3_all.get(config, {})
        h200_config = '6/6/6' if 'Shallow' in config else config
        h200 = h200_all.get(h200_config, {})

        w_iters = [wse3.get(g, {}).get('iterations', 0) for g in plot_sizes]
        h_iters = [h200.get(g, {}).get('iterations', 0) for g in plot_sizes]

        ax.bar(x + offset * bar_width, h_iters, bar_width,
               label=f'GH200 {h200_config}', color=ch, edgecolor='black', linewidth=0.5)
        ax.bar(x + (offset + 1) * bar_width, w_iters, bar_width,
               label=f'WSE-3 {config}', color=cw, edgecolor='black', linewidth=0.5)
        offset += 2

    ax.set_xlabel('Problem Size (nx × ny)', fontsize=14)
    ax.set_ylabel('Iterations to Converge', fontsize=14)
    ax.set_title('V-Cycle Iterations to Convergence', fontsize=16)
    ax.set_xticks(x + 2.5 * bar_width)
    ax.set_xticklabels(plot_sizes, fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    ax.tick_params(axis='y', labelsize=12)

    # Annotate H200 bars that hit max_iter=100
    for i, g in enumerate(plot_sizes):
        for config, h200_config in [('6/6/6', '6/6/6'), ('6/6/100', '6/6/100')]:
            h = h200_all.get(h200_config, {}).get(g, {})
            if h.get('iterations') == 100:
                # Find the bar position
                cfg_idx = configs.index(config) if config in configs else -1
                if cfg_idx >= 0:
                    bar_x = x[i] + cfg_idx * 2 * bar_width
                    ax.annotate('max', (bar_x, 100), textcoords="offset points",
                                xytext=(0, 5), fontsize=7, ha='center', color='red')

    plt.tight_layout()
    outpath = os.path.join(output_dir, 'iterations_comparison.png')
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {outpath}")


def main():
    # Parse all WSE-3 data
    wse3_all = {}
    for config_label, filename in WSE3_CONFIGS:
        filepath = os.path.join(BASE_DIR, filename)
        wse3_all[config_label] = parse_wse3_responses(filepath)

    # Parse H200 CSV
    csv_path = os.path.join(SCRIPT_DIR, 'h200_vs_cs3_feb7.csv')
    h200_all = parse_h200_csv(csv_path)

    # Print table
    print_tts_table(wse3_all, h200_all)

    # Print summary for rebuttal
    print("\n" + "=" * 80)
    print("KEY FINDINGS FOR REBUTTAL")
    print("=" * 80)
    print("\nTime-to-Solution speedup (TTS = iterations × avg_time_per_vcycle):")
    print("Note: H200 iterations capped at max_iter=100 for 64x64+, may not have converged.\n")

    for config in ['6/6/6', '6/6/6(Shallow)', '6/6/100']:
        wse3 = wse3_all.get(config, {})
        h200_config = '6/6/6' if 'Shallow' in config else config
        h200 = h200_all.get(h200_config, {})

        print(f"  Config {config}:")
        for g in ['128x128', '256x256', '512x512']:
            w = wse3.get(g, {})
            h = h200.get(g, {})
            if w.get('tts_s') and h.get('tts_s') and w['tts_s'] > 0:
                speedup = h['tts_s'] / w['tts_s']
                print(f"    {g}: WSE-3 {w['iterations']} iters in {w['tts_s']:.6f}s, "
                      f"GH200 {h['iterations']} iters in {h['tts_s']:.6f}s, "
                      f"TTS speedup = {speedup:.1f}x")
        print()

    # Generate plots
    plot_tts_comparison(wse3_all, h200_all, SCRIPT_DIR)
    plot_iterations_comparison(wse3_all, h200_all, SCRIPT_DIR)


if __name__ == '__main__':
    main()
