#!/usr/bin/env python3
"""
Fixed Tolerance Analysis: At which iteration does each problem size
cross a FIXED absolute tolerance of 1e-5?

tolerance rather than the relative tolerance (abs_tol * ||b||_2) currently used.

Usage:
  python fixed_tolerance_analysis.py
"""

import os
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(SCRIPT_DIR, '..')
RESPONSES_DIR = os.path.join(BASE_DIR, 'build')  # all_responses_*.txt now live under build/

FIXED_TOLERANCE = 1e-5

GRID_SIZES = ['4x4', '8x8', '16x16', '32x32', '64x64', '128x128', '256x256', '512x512']

WSE3_CONFIGS = [
    ('6/6/100', 'all_responses_6_6_100.txt'),
    ('4/4/6',   'all_responses_4_4_6.txt'),
    ('6/6/6',   'all_responses_6_6_6.txt'),
    ('4/4/100', 'all_responses_4_4_100.txt'),
    ('6/6/6(Shallow)', 'all_responses_6_6_6_shallow.txt'),
]


def parse_rho_and_timing(filepath):
    """Parse rho history and avg_vcycle_time from all_responses_*.txt."""
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

        # Extract rho history
        rho_history = []
        rho_section = re.search(
            r'Rho values after each iteration.*?'
            r'Total iterations performed:\s*\d+\s*\n'
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

        # Extract avg V-cycle time
        m_avg = re.search(
            r'(?:Wall time per V-cycle \(total / iterations\)'
            r'|1-V cycle time\(Average\)\s*\(us\[cycles\]\)'
            r'|1st V-cycle time \(measured\)'
            r'|Avg V-cycle time \(no conv\))'
            r'\s*:\s*([\d.]+)\s*us',
            block,
        )
        avg_vcycle_us = float(m_avg.group(1)) if m_avg else None

        # Extract relative tolerance used
        m_tol = re.search(r'Tolerance\s*=\s*([\d.eE+\-]+)', block)
        rel_tolerance = float(m_tol.group(1)) if m_tol else None

        # Extract device iterations (actual iterations performed)
        m_iter = re.search(r'Device iterations\s*:\s*(\d+)', block)
        device_iters = int(m_iter.group(1)) if m_iter else len(rho_history)

        if rho_history and avg_vcycle_us:
            results[grid_label] = {
                'rho_history': rho_history,
                'avg_vcycle_us': avg_vcycle_us,
                'rel_tolerance': rel_tolerance,
                'device_iters': device_iters,
            }

    return results


def find_iteration_at_tolerance(rho_history, tolerance):
    """Find the first iteration where |rho| <= tolerance. Returns None if never reached."""
    for i, rho in enumerate(rho_history):
        if rho <= tolerance:
            return i + 1  # 1-indexed
    return None


def main():
    # Parse all configs
    all_data = {}
    for config_label, filename in WSE3_CONFIGS:
        filepath = os.path.join(RESPONSES_DIR, filename)
        all_data[config_label] = parse_rho_and_timing(filepath)

    # H200 CSV data for comparison
    import csv
    csv_path = os.path.join(SCRIPT_DIR, 'h200_vs_cs3_feb7.csv')
    h200_data = {}
    if os.path.exists(csv_path):
        with open(csv_path, 'r') as f:
            lines = f.readlines()
        # H200 configs: col (1,2,3)=6/6/100, (4,5,6)=4/4/6, (7,8,9)=6/6/6, (10,11,12)=4/4/100
        h200_cols = {'6/6/100': (1, 2, 3), '4/4/6': (4, 5, 6), '6/6/6': (7, 8, 9), '4/4/100': (10, 11, 12)}
        grid_labels = ['4x4', '8x8', '16x16', '32x32', '64x64', '128x128', '256x256', '512x512']
        for line_idx, line in enumerate(lines[12:20]):
            row = next(csv.reader([line]))
            grid = grid_labels[line_idx]
            for cfg, (col_tts, col_iter, col_vcyc) in h200_cols.items():
                try:
                    tts_s = float(row[col_tts]) if row[col_tts].strip() else None
                    iters = int(float(row[col_iter])) if row[col_iter].strip() else None
                    vcyc_s = float(row[col_vcyc]) if row[col_vcyc].strip() else None
                except (ValueError, IndexError):
                    tts_s = iters = vcyc_s = None
                if tts_s is not None:
                    h200_data.setdefault(cfg, {})[grid] = {
                        'tts_s': tts_s, 'iterations': iters, 'per_vcycle_s': vcyc_s
                    }

    # ============================================================
    # Table 1: Fixed tolerance analysis
    # ============================================================
    print("=" * 130)
    print(f"FIXED TOLERANCE ANALYSIS: |rho|_inf <= {FIXED_TOLERANCE:.0e}")
    print("=" * 130)
    print(f"\nComparison: relative tolerance (current) vs fixed absolute tolerance ({FIXED_TOLERANCE:.0e})")

    for config_label, _ in WSE3_CONFIGS:
        config_data = all_data.get(config_label, {})
        if not config_data:
            continue

        print(f"\n--- Config: {config_label} ---")
        print(f"{'Grid':<12} {'Rel Tol':>12} {'Rel Iters':>10} {'Rel TTS(us)':>14} "
              f"{'Fix Iters':>10} {'Fix TTS(us)':>14} {'Saved Iters':>12} {'Saved %':>8}")
        print("-" * 100)

        for grid in GRID_SIZES:
            d = config_data.get(grid)
            if not d:
                continue

            rel_iters = d['device_iters']
            rel_tts_us = rel_iters * d['avg_vcycle_us']
            rel_tol = d['rel_tolerance']

            fix_iters = find_iteration_at_tolerance(d['rho_history'], FIXED_TOLERANCE)

            if fix_iters is not None:
                fix_tts_us = fix_iters * d['avg_vcycle_us']
                saved = rel_iters - fix_iters
                saved_pct = f"{100.0 * saved / rel_iters:.1f}%"
            else:
                fix_tts_us = None
                saved = None
                saved_pct = "N/A"

            rel_tol_str = f"{rel_tol:.2e}" if rel_tol else "-"
            fix_iters_str = str(fix_iters) if fix_iters else f">{len(d['rho_history'])}"
            fix_tts_str = f"{fix_tts_us:.3f}" if fix_tts_us else "-"
            saved_str = str(saved) if saved is not None else "-"

            print(f"{grid:<12} {rel_tol_str:>12} {rel_iters:>10} {rel_tts_us:>14.3f} "
                  f"{fix_iters_str:>10} {fix_tts_str:>14} {saved_str:>12} {saved_pct:>8}")

    # ============================================================
    # Table 2: TTS speedup at FIXED tolerance
    # ============================================================
    print("\n" + "=" * 130)
    print(f"TIME-TO-SOLUTION SPEEDUP AT FIXED TOLERANCE (|rho|_inf <= {FIXED_TOLERANCE:.0e})")
    print("=" * 130)
    print("\nNote: H200 iteration data is total TTS (may include iterations past convergence).")
    print("WSE-3 TTS at fixed tolerance = iterations_to_fixed_tol × avg_vcycle_time\n")

    for config_label, _ in WSE3_CONFIGS:
        config_data = all_data.get(config_label, {})
        h200_cfg = '6/6/6' if 'Shallow' in config_label else config_label
        h200_cfg_data = h200_data.get(h200_cfg, {})

        if not config_data:
            continue

        print(f"--- Config: {config_label} ---")
        print(f"{'Grid':<12} {'WSE3 Fix Iters':>15} {'WSE3 Fix TTS(s)':>16} "
              f"{'H200 TTS(s)':>14} {'Fix TTS Speedup':>16} {'Rel TTS Speedup':>16}")
        print("-" * 95)

        for grid in GRID_SIZES:
            d = config_data.get(grid)
            h = h200_cfg_data.get(grid)
            if not d:
                continue

            fix_iters = find_iteration_at_tolerance(d['rho_history'], FIXED_TOLERANCE)
            rel_iters = d['device_iters']
            rel_tts_s = rel_iters * d['avg_vcycle_us'] * 1e-6

            if fix_iters is not None:
                fix_tts_s = fix_iters * d['avg_vcycle_us'] * 1e-6
                fix_iters_str = str(fix_iters)
                fix_tts_str = f"{fix_tts_s:.6f}"
            else:
                fix_tts_s = None
                fix_iters_str = f">{len(d['rho_history'])}"
                fix_tts_str = "-"

            h_tts_str = f"{h['tts_s']:.6f}" if h else "-"

            if fix_tts_s and h and h['tts_s'] > 0:
                fix_speedup = f"{h['tts_s'] / fix_tts_s:.2f}x"
            else:
                fix_speedup = "-"

            if h and h['tts_s'] > 0 and rel_tts_s > 0:
                rel_speedup = f"{h['tts_s'] / rel_tts_s:.2f}x"
            else:
                rel_speedup = "-"

            print(f"{grid:<12} {fix_iters_str:>15} {fix_tts_str:>16} "
                  f"{h_tts_str:>14} {fix_speedup:>16} {rel_speedup:>16}")
        print()

    # ============================================================
    # Table 3: Rho at each iteration for 6/6/6 config (the main one)
    # ============================================================
    print("\n" + "=" * 130)
    print("DETAILED RHO HISTORY: 6/6/6 config — showing when fixed tolerance (1e-5) is crossed")
    print("=" * 130)

    config_data = all_data.get('6/6/6', {})
    for grid in ['64x64', '128x128', '256x256', '512x512']:
        d = config_data.get(grid)
        if not d:
            continue

        print(f"\n  {grid} (rel_tol={d['rel_tolerance']:.2e}):")
        print(f"  {'Iter':>5} {'|rho|_inf':>14} {'<= 1e-5?':>10} {'<= rel_tol?':>12}")
        print(f"  {'-'*45}")

        for i, rho in enumerate(d['rho_history']):
            fix_check = "YES <---" if rho <= FIXED_TOLERANCE else ""
            rel_check = "YES" if d['rel_tolerance'] and rho <= d['rel_tolerance'] else ""
            print(f"  {i+1:>5} {rho:>14.6e} {fix_check:>10} {rel_check:>12}")

    # ============================================================
    # Plot: Fixed vs Relative tolerance convergence
    # ============================================================
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    key_sizes = ['128x128', '256x256', '512x512']

    config_colors = {
        '6/6/100': '#F44336', '4/4/6': '#FF9800', '6/6/6': '#2196F3',
        '4/4/100': '#9C27B0', '6/6/6(Shallow)': '#4CAF50',
    }
    config_markers = {
        '6/6/100': 'o', '4/4/6': 's', '6/6/6': '^',
        '4/4/100': 'D', '6/6/6(Shallow)': 'v',
    }

    for idx, grid in enumerate(key_sizes):
        ax = axes[idx]

        for config_label, _ in WSE3_CONFIGS:
            d = all_data.get(config_label, {}).get(grid)
            if not d or not d['rho_history']:
                continue

            rho = d['rho_history']
            iters = list(range(1, len(rho) + 1))
            ax.semilogy(iters, rho, config_markers[config_label] + '-',
                        color=config_colors[config_label],
                        label=config_label, linewidth=2, markersize=6)


        # Draw fixed tolerance line
        ax.axhline(y=FIXED_TOLERANCE, color='red', linestyle='-', linewidth=2,
                    alpha=0.7, label=f'Fixed tol (1e-5)')

        # Draw relative tolerance line (from 6/6/6 config) with per-subplot label
        d_666 = all_data.get('6/6/6', {}).get(grid)
        if d_666 and d_666['rel_tolerance']:
            ax.axhline(y=d_666['rel_tolerance'], color='gray', linestyle='--',
                        linewidth=1.5, alpha=0.7)
            ax.text(0.98, d_666['rel_tolerance'] * 1.8, f"rel tol = {d_666['rel_tolerance']:.1e}",
                    transform=ax.get_yaxis_transform(), fontsize=9, color='gray',
                    ha='right', va='bottom')

        size_num = grid.split('x')[0]
        ax.set_title(f'{size_num}³ domain ({grid} PEs)', fontsize=14)
        ax.set_xlabel('V-Cycle Iteration', fontsize=13)
        if idx == 0:
            ax.set_ylabel('Residual |rho|_inf', fontsize=13)
        ax.grid(alpha=0.3)
        ax.tick_params(axis='both', labelsize=11)

    # Build legend from first subplot (configs + fixed tol line, no rel tol)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right', fontsize=10,
               bbox_to_anchor=(1.13, 0.5))

    plt.suptitle('Convergence: Fixed Tolerance (1e-5) vs Relative Tolerance',
                 fontsize=15)
    plt.tight_layout(rect=[0, 0, 0.88, 0.90])
    outpath = os.path.join(SCRIPT_DIR, 'fixed_tolerance_convergence.png')
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {outpath}")


if __name__ == '__main__':
    main()
