#!/usr/bin/env python3
"""
Plot HPGMG benchmark speedup (WSE-3 over H200) from CSV data.
Reads CSV via command line; outputs publication-quality figures.
"""

import argparse
import csv
import os
import re

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless/paper generation
import matplotlib.pyplot as plt
import numpy as np


# Publication-quality parameters
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'


def parse_csv(csv_path: str):
    """
    Parse HPGMG benchmark CSV.
    Returns (grid_sizes, data_dict) where data_dict maps config name -> list of speedups.
    """
    # Display order in legend and bars
    configs = ['6/6/100', '4/4/100', '4/4/6', '6/6/6']
    # CSV column order (last 4 columns): over(6/6/100), over(4/4/6), 6/6/6, 4/4/100
    csv_col_order = ['6/6/100', '4/4/6', '6/6/6', '4/4/100']
    grid_sizes = []
    values = {c: [] for c in configs}

    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        rows = list(reader)

    # Find header row containing speedup column names
    header_row_idx = None
    for i, row in enumerate(rows):
        if len(row) >= 4:
            # Check last columns for config-like headers
            tail = [c.strip().lower() for c in row[-4:] if c]
            if '6/6/100' in str(tail) or '4/4/6' in str(tail):
                header_row_idx = i
                break

    # Grid size is in column 0 (A)
    GRID_COL = 0
    # Speedup columns are the last 4
    N_SPEEDUP = 4

    for i in range((header_row_idx or 0) + 1, len(rows)):
        row = rows[i]
        if len(row) <= GRID_COL:
            continue
        gs = str(row[GRID_COL]).strip()
        # Match grid size pattern: NxN or NxxN (typo)
        if not re.match(r'^\d+x+\d+$', gs):
            continue
        gs = re.sub(r'x+', 'x', gs)  # normalize 64xx64 -> 64x64
        if len(row) < N_SPEEDUP:
            continue
        speedups = []
        for j in range(-N_SPEEDUP, 0):
            try:
                v = float(row[j].strip().replace(',', ''))
            except (ValueError, IndexError):
                v = None
            speedups.append(v)
        if all(v is not None and v > 0 for v in speedups):
            grid_sizes.append(gs)
            for c in configs:
                idx = csv_col_order.index(c)
                values[c].append(speedups[idx])

    return grid_sizes, values


def plot_hpgmg_speedup_bar(csv_path: str, out_path: str = 'hpgmg_speedup_barplot.pdf'):
    grid_sizes, data = parse_csv(csv_path)
    if not grid_sizes:
        raise ValueError(f"No valid speedup data found in {csv_path}")

    x = np.arange(len(grid_sizes))
    width = 0.2
    fig, ax = plt.subplots(figsize=(8, 5))

    # Only assign colors/offsets to configs that actually plot bars
    bar_configs = ['6/6/100', '4/4/100', '4/4/6', '6/6/6']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    offsets = np.linspace(-1.5 * width, 1.5 * width, len(bar_configs))

    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    handles = []
    labels = []

    # Bar groups for the actual configs
    for i, config in enumerate(bar_configs):
        values = data.get(config, [])
        if values:
            offset = offsets[i]
            ax.bar(x + offset, values, width, color=colors[i],
                   edgecolor='black', alpha=0.85, linewidth=0.5)
            # Color patch without box (edgecolor='none') for legend
            handles.append(Patch(facecolor=colors[i], edgecolor='none', alpha=0.85))
            labels.append(config)

    # Add GH200 baseline and --- line for legend
    ax.axhline(y=1, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
    handles.append(Line2D([0], [0], color= "black", linestyle='--', linewidth=1))
    labels.append('Baseline (1.0×)')

    ax.set_xlabel('Grid sizes')
    ax.set_ylabel('Relative speedup over GH200')
    ax.set_title('GLOW VS HPGMG')
    ax.set_xticks(x)
    ax.set_xticklabels(grid_sizes)
    ax.legend(
        handles=handles,
        labels=labels,
        title='Configuration (pre/post/coarse)',
        loc='upper center',
        bbox_to_anchor=(0.5, 1.0),
        frameon=True,
        fancybox=True,
        ncol=1
    )
    ax.set_axisbelow(True)
    ax.grid(True, axis='y', linestyle='--', alpha=0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Ensure y-axis starts at 0 for clearer bar comparison
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(out_path, format='pdf', dpi=300, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.close()


def main():
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    default_csv = os.path.join(_script_dir, 'h200_vs_cs3_feb6.csv')
    ap = argparse.ArgumentParser(description='Plot HPGMG speedup from CSV')
    ap.add_argument('csv', nargs='?', default=default_csv,
                    help='Input CSV path (default: h200_vs_cs3_feb6.csv in script dir)')
    ap.add_argument('-o', '--output', default='hpgmg_speedup_barplot.pdf',
                    help='Output figure path (default: hpgmg_speedup_barplot.pdf)')
    args = ap.parse_args()
    plot_hpgmg_speedup_bar(args.csv, args.output)


if __name__ == '__main__':
    main()
