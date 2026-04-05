#!/usr/bin/env python3
"""Spider (radar) plot for GMG V-cycle solver on WSE-3.

7-axis radar chart showing hardware utilization across compute, memory,
network, and algorithmic dimensions for a single problem configuration.

Axes (all 0-100%, higher = better utilization):
  1. Compute Efficiency    — Per-PE achieved FLOP/s / peak 1.75 GFLOP/s
  2. Memory Bandwidth      — Per-PE achieved SRAM BW / peak 28 GB/s
  3. Network Injection BW  — Achieved fabric BW / peak 3.5 GB/s (CE-router)
  4. PE Utilization         — Time-weighted active PE fraction across MG levels
  5. SRAM Usage             — Code + data footprint / 48 KB per PE
  6. Arithmetic Intensity   — AI / (2 x ridge point), normalized
  7. Compute Fraction       — Compute time / total SpMV time (inverse of comm overhead)

Usage:
    python plots/spider_plot.py --size 256
    python plots/spider_plot.py --size 512 --config P4_P4_B100
"""

import re
import os
import sys
import glob
import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from math import pi

# ── WSE-3 machine parameters ──────────────────────────────────────────────
PE_FREQ_MHZ = 875
PE_PEAK_GFLOPS = 1.75          # 875 MHz x 1 FMAC/cycle x 2 FLOPs
PE_PEAK_BW_GBS = 28.0          # 14 GB/s read + 14 GB/s write
RIDGE_POINT = PE_PEAK_GFLOPS / PE_PEAK_BW_GBS  # 0.0625 FLOP/Byte
SRAM_PER_PE_BYTES = 48 * 1024  # 48 KB
FABRIC_BW_GBS = 3.5            # CE-router injection: 1 wavelet/cycle x 4B x 875 MHz
TOTAL_WSE3_PES = 893_064

# Calibrated from 256^3 roofline (508 FLOPs per z-element per V-cycle)
FLOPS_PER_Z_ELEM = 508
CONSTANT_AI = 0.103
BLOCK_SIZES = {4: 4, 8: 8, 16: 16, 32: 32, 64: 64, 128: 128, 256: 256, 512: 256}


# ── Parsing ────────────────────────────────────────────────────────────────

def _ffloat(pattern, text, default=None):
    m = re.search(pattern, text)
    return float(m.group(1)) if m else default

def _fint(pattern, text, default=None):
    m = re.search(pattern, text)
    return int(m.group(1).replace(',', '')) if m else default


def parse_response(filepath):
    with open(filepath, 'r') as f:
        return parse_text(f.read())

def parse_text(text):
    d = {}
    m = re.search(r'HeightxWidthxZDim\s*:\s*(\d+)x(\d+)x(\d+)', text)
    if m:
        d['size'] = int(m.group(1))
    d['levels']        = _fint(r'Levels\s*:\s*(\d+)', text)
    d['iterations']    = _fint(r'Device iterations\s*:\s*(\d+)', text)
    d['max_iterations'] = _fint(r'Max iterations\s*:\s*(\d+)', text)
    d['block_size']    = _fint(r'Block size\s*:\s*(\d+)', text)
    d['pre_iter']      = _fint(r'pre_iter=(\d+)', text, 6)
    d['post_iter']     = _fint(r'post_iter=(\d+)', text, 6)
    d['bottom_iter']   = _fint(r'bottom_iter=(\d+)', text, 6)

    # Roofline (may be absent in older runs)
    d['compute_util_pct'] = _ffloat(r'% of per-PE peak:\s*([\d.]+)%', text)
    d['total_flops_pe']   = _fint(r'Total FLOPs \(PE 0,0\):\s*([\d,]+)', text)
    d['arith_intensity']  = _ffloat(r'Arith\. Intensity \(PE 0,0\):\s*([\d.]+)', text)

    # Solver time
    d['ops_time_us'] = _ffloat(
        r'(?:Sum of per-level totals|Total Upper bound V-cycle time).*?:\s*([\d.]+)\s*us',
        text,
    )
    d['solver_time_us'] = _ffloat(
        r'(?:Total solver wall time|Total Solver time).*?:\s*([\d.]+)\s*us',
        text,
    )
    if d['solver_time_us'] is None:
        d['solver_time_us'] = _ffloat(
            r'Total V-cycle time \(Kernel Launch.*?:\s*([\d.]+)\s*us', text)

    # Average V-cycle time
    d['avg_vcycle_us'] = _ffloat(
        r'(?:Wall time per V-cycle \(total / iterations\)'
        r'|1-V.cycle time'
        r'|1st V-cycle time \(measured\)'
        r'|Avg V-cycle time \(no conv\))'
        r'.*?:\s*([\d.]+)\s*us',
        text,
    )
    if d['avg_vcycle_us'] is None and d.get('solver_time_us') and d.get('iterations'):
        d['avg_vcycle_us'] = d['solver_time_us'] / d['iterations']

    # Fine-level PEs
    d['fine_pes'] = _fint(r'Active PEs \(fine level\):\s*([\d,]+)', text)
    if d['fine_pes'] is None:
        d['fine_pes'] = _fint(r'System \(allocated ([\d,]+) PEs\)', text)
    if d['fine_pes'] is None and d.get('size'):
        d['fine_pes'] = d['size'] ** 2

    # Per-level timing (main timing table only)
    main_table = re.search(
        r'Time per operation and level.*?\n(.*?)(?=\n\s*7-pt Stencil|\n\n\n)',
        text, re.DOTALL)
    level_times = {}
    if main_table:
        for rm in re.finditer(
                r'\|\s+(\d+)\s+\|.*?([\d.]+)us\(\s*\d+\)\s*\|\s*$',
                main_table.group(1), re.MULTILINE):
            level_times[int(rm.group(1))] = float(rm.group(2))
    d['level_times'] = level_times

    # Stencil compute vs communication (total row)
    stencil = re.search(
        r'7-pt Stencil.*?\n(.*?)(?:\n\n|\nInterpolation)', text, re.DOTALL)
    if stencil:
        tr = re.search(
            r'\|\s*total\s*\|'
            r'\s*([\d.]+)us\(\s*\d+\)\s*\|'
            r'\s*([\d.]+)us\(\s*\d+\)\s*\|'
            r'\s*([\d.]+)us\(\s*\d+\)\s*\|',
            stencil.group(1))
        if tr:
            d['spmv_total_us']   = float(tr.group(1))
            d['spmv_comm_us']    = float(tr.group(2))
            d['spmv_compute_us'] = float(tr.group(3))

    # SRAM
    sm = re.search(r'Total accounted:\s+(\d+)\s+bytes', text)
    if sm:
        d['sram_bytes'] = int(sm.group(1))

    return d


# ── Fabric traffic estimation ─────────────────────────────────────────────

def estimate_fabric_bytes_per_vcycle(size, levels, block_size, pre, post, bottom):
    """Estimate total fabric (network) bytes transferred per PE per V-cycle.

    Each stencil (SpMV) call sends/receives one z-column halo to/from each
    of 4 neighbors:  bytes_per_call = 4 directions x 2 (send+recv) x nz x 4B
    """
    total = 0
    for l in range(levels):
        nz = max(2, block_size >> l)
        bytes_per_spmv = 4 * 2 * nz * 4  # 4 dirs, send+recv, nz floats

        if l == 0:
            # pre-smooth + post-smooth + residual + convergence check
            n_spmv = pre + post + 1 + 1
        elif l == levels - 1:
            # bottom level: only smooth iterations
            n_spmv = bottom
        else:
            # pre-smooth + post-smooth + residual
            n_spmv = pre + post + 1

        total += n_spmv * bytes_per_spmv
    return total


# ── Spider metrics (7 axes) ───────────────────────────────────────────────

def compute_spider_metrics(d):
    s = {}
    size = d.get('size', 0)
    levels = d.get('levels', 1)
    block_size = d.get('block_size') or BLOCK_SIZES.get(size, size)
    avg_us = d.get('avg_vcycle_us', 1)
    iters = d.get('iterations', 1)
    pre  = d.get('pre_iter', 6)
    post = d.get('post_iter', 6)
    bot  = d.get('bottom_iter', 6)

    # Estimate FLOPs per V-cycle
    if d.get('total_flops_pe') and iters:
        flops_per_v = d['total_flops_pe'] / iters
    else:
        flops_per_v = FLOPS_PER_Z_ELEM * block_size

    ai = d.get('arith_intensity') or CONSTANT_AI

    # 1. Compute Efficiency (per-PE FLOP/s / peak)
    if d.get('compute_util_pct') is not None:
        s['Compute\nEfficiency'] = d['compute_util_pct']
    else:
        achieved = flops_per_v / (avg_us * 1e-6)
        s['Compute\nEfficiency'] = (achieved / (PE_PEAK_GFLOPS * 1e9)) * 100

    # 2. Memory Bandwidth (per-PE SRAM BW / peak 28 GB/s)
    mem_per_v = flops_per_v / ai  # bytes
    bw_gbs = (mem_per_v / avg_us) * 1e-3  # B/us -> GB/s
    s['Memory\nBandwidth'] = min(100.0, (bw_gbs / PE_PEAK_BW_GBS) * 100)

    # 3. Network Injection BW (fabric BW / peak 3.5 GB/s)
    fabric_bytes = estimate_fabric_bytes_per_vcycle(
        size, levels, block_size, pre, post, bot)
    comm_us = d.get('spmv_comm_us', 0)
    if comm_us and iters:
        comm_per_v = comm_us / iters
        fabric_bw_gbs = (fabric_bytes / comm_per_v) * 1e-3  # B/us -> GB/s
        s['Network\nInjection BW'] = min(100.0, (fabric_bw_gbs / FABRIC_BW_GBS) * 100)
    else:
        s['Network\nInjection BW'] = 0

    # 4. PE Utilization (time-weighted active fraction)
    fine_pes = d.get('fine_pes', size * size)
    lt = d.get('level_times', {})
    if lt and fine_pes > 0:
        num = sum((min(1.0, max(1, (size // (2**l))**2) / fine_pes) * t)
                  for l, t in lt.items())
        den = sum(lt.values())
        s['PE\nUtilization'] = (num / den * 100) if den else 100
    else:
        fracs = [min(1.0, 1.0 / (4**l)) for l in range(levels)]
        s['PE\nUtilization'] = (sum(fracs) / len(fracs)) * 100

    # 5. SRAM Usage (code + data / 48 KB)
    if d.get('sram_bytes'):
        s['SRAM\nUsage'] = (d['sram_bytes'] / SRAM_PER_PE_BYTES) * 100
    else:
        est = 23344 + block_size * 54
        s['SRAM\nUsage'] = min(100, (est / SRAM_PER_PE_BYTES) * 100)

    # 6. Arithmetic Intensity (AI / (2 x ridge point), capped at 100%)
    ratio = ai / RIDGE_POINT  # ~1.65 for 7-pt stencil
    s['Arithmetic\nIntensity'] = min(100, ratio / 2 * 100)

    # 7. Compute Fraction (compute / total SpMV — inverse of comm overhead)
    if d.get('spmv_compute_us') and d.get('spmv_total_us'):
        s['Compute\nFraction'] = (d['spmv_compute_us'] / d['spmv_total_us']) * 100
    else:
        s['Compute\nFraction'] = 30

    return s


# ── Plotting ───────────────────────────────────────────────────────────────

def make_spider_plot(spider, label, raw, output_path, title=None):
    categories = list(spider.keys())
    N = len(categories)
    angles = [n / N * 2 * pi for n in range(N)]
    angles += angles[:1]
    values = [spider[c] for c in categories] + [spider[categories[0]]]

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)

    color = '#1565C0'
    ax.plot(angles, values, 'o-', linewidth=2.8, color=color, markersize=9,
            zorder=3, markeredgecolor='white', markeredgewidth=1.5)
    ax.fill(angles, values, alpha=0.15, color=color)

    # Value labels
    for angle, val, cat in zip(angles[:-1], values[:-1], categories):
        if 0 < angle < pi:
            ha = 'left'
        elif angle > pi:
            ha = 'right'
        else:
            ha = 'center'
        ax.text(angle, val + 7, f'{val:.1f}%', ha=ha, va='center',
                fontsize=12, fontweight='bold', color='#333333',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                          edgecolor='#CCCCCC', alpha=0.85))

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=11, fontweight='bold',
                       multialignment='center')
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'],
                       size=9, color='#999999')
    ax.set_ylim(0, 115)
    ax.grid(color='#CCCCCC', linestyle='--', linewidth=0.6, alpha=0.7)
    ax.spines['polar'].set_visible(False)

    if title:
        ax.set_title(title, size=15, fontweight='bold', pad=35, color='#333333')

    # Info box
    size = raw.get('size', '?')
    levels = raw.get('levels', '?')
    iters = raw.get('iterations', '?')
    vcycle = raw.get('avg_vcycle_us', 0)
    pes = raw.get('fine_pes', 0)
    info_lines = [
        f'{size}\u00b3 grid, {levels} levels, {iters} V-cycles',
        f'{vcycle:.0f} \u00b5s/V-cycle, {pes:,} PEs',
        f'Peak: {PE_PEAK_GFLOPS} GFLOP/s, {PE_PEAK_BW_GBS} GB/s SRAM, {FABRIC_BW_GBS} GB/s fabric'
    ]
    ax.text(0.98, -0.06, '\n'.join(info_lines), transform=ax.transAxes,
            fontsize=9, ha='right', va='top', color='#666666',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#F5F5F5',
                      edgecolor='#DDDDDD'))

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved: {output_path}")


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Spider plot — GMG V-cycle hardware utilization')
    parser.add_argument('--size', type=int, default=256,
                        help='Problem size (default: 256)')
    parser.add_argument('--config', default='P6_P6_B6',
                        help='Config suffix (default: P6_P6_B6)')
    parser.add_argument('--output', default=None)
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(base_dir)

    # Find response file (out_dir_* live under build/ since the scripts/output split)
    pattern = os.path.join(parent_dir, 'build',
                           f'out_dir_S{args.size}x_L*_M100_{args.config}/response.txt')
    files = glob.glob(pattern)
    if not files:
        # Back-compat: try legacy location (pre-build/ reorg)
        files = glob.glob(os.path.join(parent_dir,
                                       f'out_dir_S{args.size}x_L*_M100_{args.config}/response.txt'))
    if not files:
        print(f"No response file for size={args.size}, config={args.config}")
        sys.exit(1)

    d = parse_response(files[0])

    # Fallback to all_responses_roofline.txt if response is truncated
    if d.get('avg_vcycle_us') is None:
        agg = os.path.join(parent_dir, 'build', 'all_responses_roofline.txt')
        if os.path.exists(agg):
            dirname = os.path.basename(os.path.dirname(files[0]))
            with open(agg) as f:
                agg_text = f.read()
            m = re.search(
                rf'Output directory:\s*{re.escape(dirname)}\n(.*?)(?=\nOutput directory:|\Z)',
                agg_text, re.DOTALL)
            if m:
                d = parse_text(f"Output directory: {dirname}\n" + m.group(1))

    spider = compute_spider_metrics(d)

    out = args.output or os.path.join(base_dir, 'spider_plot.png')
    make_spider_plot(
        spider, f"{args.size}\u00b3", d, out,
        title=f'GMG V-Cycle \u2014 Hardware Utilization ({args.size}\u00b3, WSE-3)')

    # Summary
    print(f"\nMetrics for {args.size}\u00b3:")
    print("-" * 45)
    peaks = {
        'Compute\nEfficiency':   f'peak {PE_PEAK_GFLOPS} GFLOP/s',
        'Memory\nBandwidth':     f'peak {PE_PEAK_BW_GBS} GB/s',
        'Network\nInjection BW': f'peak {FABRIC_BW_GBS} GB/s',
        'PE\nUtilization':       f'{d.get("fine_pes",0):,} fine PEs',
        'SRAM\nUsage':           f'{SRAM_PER_PE_BYTES//1024} KB',
        'Arithmetic\nIntensity': f'ridge={RIDGE_POINT:.4f}',
        'Compute\nFraction':     'compute/SpMV',
    }
    for cat, val in spider.items():
        flat = cat.replace('\n', ' ')
        ref = peaks.get(cat, '')
        print(f"  {flat:<25s} {val:6.1f}%   ({ref})")


if __name__ == '__main__':
    main()
