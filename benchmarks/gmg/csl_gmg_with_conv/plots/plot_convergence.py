#!/usr/bin/env python3
"""
Convergence Plot for GMG V-Cycle on WSE-3.

Extracts rho_history (residual per iteration) from all_responses_*.txt
and generates residual-vs-iteration convergence plots.

Produces:
  1. convergence_by_size.png  - One subplot per grid size, overlaying configs
  2. convergence_by_config.png - One subplot per config, overlaying grid sizes

Usage:
  python plot_convergence.py
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

GRID_SIZES = ['4x4', '8x8', '16x16', '32x32', '64x64', '128x128', '256x256', '512x512']

WSE3_CONFIGS = [
    ('6/6/100', 'all_responses_6_6_100.txt'),
    ('4/4/6',   'all_responses_4_4_6.txt'),
    ('6/6/6',   'all_responses_6_6_6.txt'),
    ('4/4/100', 'all_responses_4_4_100.txt'),
    ('6/6/6(Shallow)', 'all_responses_6_6_6_shallow.txt'),
]

CONFIG_COLORS = {
    '6/6/100': '#F44336',
    '4/4/6':   '#FF9800',
    '6/6/6':   '#2196F3',
    '4/4/100': '#9C27B0',
    '6/6/6(Shallow)': '#4CAF50',
}

CONFIG_MARKERS = {
    '6/6/100': 'o',
    '4/4/6':   's',
    '6/6/6':   '^',
    '4/4/100': 'D',
    '6/6/6(Shallow)': 'v',
}


def parse_rho_history(filepath):
    """Parse all_responses_*.txt to extract rho history per grid size.

    Returns dict: grid_label -> {
        'rho_history': [rho_1, rho_2, ...],
        'tolerance': float,
        'converged': bool,
    }
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

        # Extract tolerance
        m_tol = re.search(r'Tolerance\s*=\s*([\d.eE+\-]+)', block)
        tolerance = float(m_tol.group(1)) if m_tol else None

        # Extract convergence
        m_conv = re.search(r'Converged:\s*(Yes|No)', block)
        converged = m_conv.group(1) == 'Yes' if m_conv else None

        if rho_history:
            results[grid_label] = {
                'rho_history': rho_history,
                'tolerance': tolerance,
                'converged': converged,
            }

    return results


def plot_convergence_512(all_data, output_dir):
    """Single figure: 512³ with 6/6/6 config."""
    grid = '512x512'
    config_label = '6/6/6'
    data = all_data.get(config_label, {}).get(grid)
    if not data or not data['rho_history']:
        print(f"No data for {config_label} {grid}")
        return

    rho = data['rho_history']
    iters = list(range(1, len(rho) + 1))

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.semilogy(iters, rho, 'o-', color='#2196F3', linewidth=2.5, markersize=8,
                markeredgecolor='black', markeredgewidth=1, label=f'{config_label}')

    if data['tolerance']:
        ax.axhline(y=data['tolerance'], color='gray', linestyle='--',
                   linewidth=1.5, alpha=0.7)
        ax.text(max(iters) * 0.65, data['tolerance'] * 2,
                f'tolerance = {data["tolerance"]:.1e}', fontsize=11, color='gray')

    ax.set_title(f'Convergence: 512³ (6/6/6)', fontsize=16, fontweight='bold')
    ax.set_xlabel('V-Cycle Iteration', fontsize=14)
    ax.set_ylabel('Residual $||r||_\\infty$', fontsize=14)
    ax.grid(alpha=0.3)
    ax.tick_params(axis='both', labelsize=12)
    ax.set_xticks(iters)

    plt.tight_layout()
    outpath = os.path.join(output_dir, 'convergence.png')
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {outpath}")


def print_convergence_table(all_data):
    """Print convergence summary table."""
    print("=" * 100)
    print("CONVERGENCE SUMMARY")
    print("=" * 100)
    print(f"\n{'Config':<20} {'Grid':<12} {'Iters':>6} {'Final Rho':>14} {'Tolerance':>14} {'Conv':>6}")
    print("-" * 80)

    for config_label, _ in WSE3_CONFIGS:
        config_data = all_data.get(config_label, {})
        for grid in GRID_SIZES:
            data = config_data.get(grid)
            if data:
                rho = data['rho_history']
                final = f"{rho[-1]:.6e}" if rho else '-'
                tol = f"{data['tolerance']:.6e}" if data['tolerance'] else '-'
                conv = 'Yes' if data['converged'] else 'No'
                print(f"{config_label:<20} {grid:<12} {len(rho):>6} {final:>14} {tol:>14} {conv:>6}")


def main():
    # Parse all configs
    all_data = {}
    for config_label, filename in WSE3_CONFIGS:
        filepath = os.path.join(RESPONSES_DIR, filename)
        all_data[config_label] = parse_rho_history(filepath)

    # Print table
    print_convergence_table(all_data)

    # Generate plot
    plot_convergence_512(all_data, SCRIPT_DIR)

    print("\nDone!")


if __name__ == '__main__':
    main()
