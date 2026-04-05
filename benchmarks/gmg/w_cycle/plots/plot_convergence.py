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


def plot_convergence_by_size(all_data, output_dir):
    """One subplot per grid size, overlaying configs."""
    # Only plot sizes with interesting data (128+)
    plot_sizes = ['16x16', '32x32', '64x64', '128x128', '256x256', '512x512']

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for idx, grid in enumerate(plot_sizes):
        ax = axes[idx]

        for config_label, _ in WSE3_CONFIGS:
            data = all_data.get(config_label, {}).get(grid)
            if data and data['rho_history']:
                rho = data['rho_history']
                iters = list(range(1, len(rho) + 1))
                ax.semilogy(iters, rho, CONFIG_MARKERS[config_label] + '-',
                            color=CONFIG_COLORS[config_label],
                            label=config_label, linewidth=2, markersize=6)

                # Draw tolerance line
                if data['tolerance']:
                    ax.axhline(y=data['tolerance'], color='gray', linestyle='--',
                               linewidth=1, alpha=0.7)

        ax.set_title(f'{grid} ({grid.split("x")[0]}³ domain)', fontsize=14)
        ax.set_xlabel('V-Cycle Iteration', fontsize=12)
        ax.set_ylabel('|rho|_inf (residual)', fontsize=12)
        ax.grid(alpha=0.3)
        ax.tick_params(axis='both', labelsize=10)

        if idx == 0:
            ax.legend(fontsize=9, loc='upper right')

    plt.suptitle('Convergence: Residual vs V-Cycle Iteration (WSE-3)', fontsize=16, y=1.01)
    plt.tight_layout()
    outpath = os.path.join(output_dir, 'convergence_by_size.png')
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {outpath}")


def plot_convergence_by_config(all_data, output_dir):
    """One subplot per config, overlaying grid sizes."""
    size_colors = {
        '16x16': '#E91E63', '32x32': '#FF5722', '64x64': '#FF9800',
        '128x128': '#4CAF50', '256x256': '#2196F3', '512x512': '#673AB7',
    }

    configs_to_plot = ['6/6/6', '6/6/6(Shallow)', '6/6/100', '4/4/6', '4/4/100']
    fig, axes = plt.subplots(1, len(configs_to_plot), figsize=(24, 5))

    for idx, config_label in enumerate(configs_to_plot):
        ax = axes[idx]
        config_data = all_data.get(config_label, {})

        for grid, color in size_colors.items():
            data = config_data.get(grid)
            if data and data['rho_history']:
                rho = data['rho_history']
                iters = list(range(1, len(rho) + 1))
                ax.semilogy(iters, rho, 'o-', color=color, label=grid,
                            linewidth=2, markersize=5)

        ax.set_title(f'Config: {config_label}', fontsize=14)
        ax.set_xlabel('V-Cycle Iteration', fontsize=12)
        if idx == 0:
            ax.set_ylabel('|rho|_inf (residual)', fontsize=12)
        ax.grid(alpha=0.3)
        ax.tick_params(axis='both', labelsize=10)
        ax.legend(fontsize=8, loc='upper right')

    plt.suptitle('Convergence by Configuration (WSE-3)', fontsize=16, y=1.02)
    plt.tight_layout()
    outpath = os.path.join(output_dir, 'convergence_by_config.png')
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {outpath}")


def plot_convergence_key_sizes(all_data, output_dir):
    """Publication-quality single figure: 128³, 256³, 512³ side by side."""
    key_sizes = ['128x128', '256x256', '512x512']
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    for idx, grid in enumerate(key_sizes):
        ax = axes[idx]
        for config_label, _ in WSE3_CONFIGS:
            data = all_data.get(config_label, {}).get(grid)
            if data and data['rho_history']:
                rho = data['rho_history']
                iters = list(range(1, len(rho) + 1))
                ax.semilogy(iters, rho, CONFIG_MARKERS[config_label] + '-',
                            color=CONFIG_COLORS[config_label],
                            label=config_label, linewidth=2.5, markersize=7)

                if data['tolerance']:
                    ax.axhline(y=data['tolerance'], color='gray', linestyle='--',
                               linewidth=1, alpha=0.5)
                    if idx == 0:
                        ax.text(max(iters) * 0.6, data['tolerance'] * 1.5,
                                'tolerance', fontsize=10, color='gray')

        size_num = grid.split('x')[0]
        ax.set_title(f'{size_num}³ domain ({grid} PEs)', fontsize=14)
        ax.set_xlabel('V-Cycle Iteration', fontsize=13)
        if idx == 0:
            ax.set_ylabel('Residual |rho|_inf', fontsize=13)
        ax.grid(alpha=0.3)
        ax.tick_params(axis='both', labelsize=11)

    # Single legend on right
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right', fontsize=11,
               bbox_to_anchor=(1.12, 0.5))

    plt.suptitle('Convergence of GMG V-Cycle on WSE-3', fontsize=16)
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    outpath = os.path.join(output_dir, 'convergence_key_sizes.png')
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
        filepath = os.path.join(BASE_DIR, filename)
        all_data[config_label] = parse_rho_history(filepath)

    # Print table
    print_convergence_table(all_data)

    # Generate plots
    plot_convergence_by_size(all_data, SCRIPT_DIR)
    plot_convergence_by_config(all_data, SCRIPT_DIR)
    plot_convergence_key_sizes(all_data, SCRIPT_DIR)

    print("\nDone!")


if __name__ == '__main__':
    main()
