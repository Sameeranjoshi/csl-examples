#!/usr/bin/env python3
"""Spider (radar) plot for GMG V-cycle solver on WSE-3.

5-axis pentagon showing hardware utilization for a single problem configuration.

Axes (all 0-100%, higher = better utilization):
  1. SRAM Usage         — Code + data footprint / 48 KB per PE
  2. PE Utilization     — Time-weighted active PE fraction across MG levels
  3. Compute Frac.      — Compute time / total SpMV time (inverse of comm overhead)
  4. Compute Eff.       — Per-PE achieved FLOP/s / peak 1.75 GFLOP/s
  5. Memory BW          — Per-PE achieved SRAM BW / peak 28 GB/s

Usage:
    python plots/spider_plot.py --size 512
    python plots/spider_plot.py --size 256 --config P6_P6_B6
"""

import re
import os
import sys
import glob
import argparse

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from math import pi

# ── Publication rcParams ──────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
})

# ── WSE-3 machine parameters ──────────────────────────────────────────────
PE_FREQ_MHZ = 875
PE_PEAK_GFLOPS = 1.75          # 875 MHz x 1 FMAC/cycle x 2 FLOPs
PE_PEAK_BW_GBS = 28.0          # 14 GB/s read + 14 GB/s write
SRAM_PER_PE_BYTES = 48 * 1024  # 48 KB
TOTAL_WSE3_PES = 893_064

# Fallback estimate (used only when response has no FLOP counters)
FLOPS_PER_Z_ELEM = 508
CONSTANT_AI = 0.103
BLOCK_SIZES = {4: 4, 8: 8, 16: 16, 32: 32, 64: 64, 128: 128, 256: 256, 512: 256}

# FLOP and memory traffic weights per operation (from roofline_analysis.py)
FLOP_PER_OP = {
    'fsub': 1, 'fmac': 2, 'fmul': 1, 'fadd': 1, 'fneg': 1,
    'fmov_mem': 0, 'fmov_zero': 0, 'fmax': 1,
}
MEM_TRAFFIC = {
    'fsub': 3, 'fmac': 4, 'fmul': 3, 'fadd': 3, 'fneg': 2,
    'fmov_mem': 2, 'fmov_zero': 1, 'fmax': 2,
}


# ── Parsing ────────────────────────────────────────────────────────────────

def _ffloat(pattern, text, default=None):
    m = re.search(pattern, text)
    return float(m.group(1)) if m else default

def _fint(pattern, text, default=None):
    m = re.search(pattern, text)
    return int(m.group(1).replace(',', '')) if m else default


def parse_flop_counters(text):
    """Parse per-level FLOP counters from response.txt roofline section.

    Returns (total_flops_per_vcycle, arithmetic_intensity) or (None, None).
    """
    level_pattern = re.compile(
        r'^Level (\d+)[^\n]*:\s*\n'
        r'Operation.*\n-+\n'
        r'((?:(?:FSUB|FMAC|FMUL|FADD|FNEG|FMOV_MEM|FMOV_ZERO|FMAX|FMOV32)\s+[\d,]+.*\n)*)',
        re.MULTILINE
    )
    matches = list(level_pattern.finditer(text))
    if not matches:
        return None, None

    m_iters = re.search(r'Device iterations(?:\s+to\s+converge)?\s*:\s*(\d+)', text)
    device_iters = int(m_iters.group(1)) if m_iters else 1

    total_flops = 0
    total_mem_words = 0
    for match in matches:
        ops_text = match.group(2)
        for line in ops_text.strip().split('\n'):
            parts = line.split()
            if len(parts) >= 2:
                op = parts[0].lower()
                count = int(parts[1].replace(',', ''))
                total_flops += count * FLOP_PER_OP.get(op, 0)
                total_mem_words += count * MEM_TRAFFIC.get(op, 0)

    flops_per_v = total_flops / device_iters
    mem_bytes_per_v = total_mem_words * 4 / device_iters  # f32 = 4 bytes
    ai = total_flops / (total_mem_words * 4) if total_mem_words else CONSTANT_AI
    return flops_per_v, ai


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

    # Parse actual FLOP counters from roofline section
    flops_per_v, actual_ai = parse_flop_counters(text)
    if flops_per_v is not None:
        d['flops_per_vcycle'] = flops_per_v
        d['actual_ai'] = actual_ai

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


# ── Spider metrics (5 axes, SYSTEM-LEVEL) ────────────────────────────────

def compute_spider_metrics(d):
    """System-level utilization: what fraction of the wafer's resources
    does this problem actually consume / achieve?"""
    s = {}
    size = d.get('size', 0)
    levels = d.get('levels', 1)
    block_size = d.get('block_size') or BLOCK_SIZES.get(size, size)
    avg_us = d.get('avg_vcycle_us', 1)
    iters = d.get('iterations', 1)
    fine_pes = d.get('fine_pes', size * size)

    # Per-PE FLOPs and AI
    if d.get('flops_per_vcycle'):
        flops_per_v = d['flops_per_vcycle']
    elif d.get('total_flops_pe') and iters:
        flops_per_v = d['total_flops_pe'] / iters
    else:
        flops_per_v = FLOPS_PER_Z_ELEM * block_size
    ai = d.get('actual_ai') or d.get('arith_intensity') or CONSTANT_AI

    # System peaks (allocated PEs, not full wafer)
    sys_peak_gflops = fine_pes * PE_PEAK_GFLOPS          # GFLOP/s
    sys_peak_bw_gbs = fine_pes * PE_PEAK_BW_GBS          # GB/s
    fab_bw_per_pe = 4 * 4 * PE_FREQ_MHZ * 1e-3           # GB/s per PE
    sys_peak_fab_gbs = fine_pes * fab_bw_per_pe           # GB/s

    # 1. Chip Area (PEs allocated / total wafer PEs)
    s['Chip Area'] = (fine_pes / TOTAL_WSE3_PES) * 100

    # 2. SRAM Usage (per-PE code + data / 48 KB)
    if d.get('sram_bytes'):
        s['SRAM Usage'] = (d['sram_bytes'] / SRAM_PER_PE_BYTES) * 100
    else:
        est = 23344 + block_size * 54
        s['SRAM Usage'] = min(100, (est / SRAM_PER_PE_BYTES) * 100)

    # 3. PE Utilization (time-weighted active fraction across MG levels)
    lt = d.get('level_times', {})
    if lt and fine_pes > 0:
        num = sum((min(1.0, max(1, (size // (2**l))**2) / fine_pes) * t)
                  for l, t in lt.items())
        den = sum(lt.values())
        s['PE Utilization'] = (num / den * 100) if den else 100
    else:
        fracs = [min(1.0, 1.0 / (4**l)) for l in range(levels)]
        s['PE Utilization'] = (sum(fracs) / len(fracs)) * 100

    # 4. System Compute (system achieved GFLOP/s / system peak)
    #    Fine-level achieved × fine_pes (PE(0,0) is representative)
    pe_achieved = flops_per_v / (avg_us * 1e-6)  # FLOP/s
    sys_achieved_gflops = pe_achieved * 1e-9  # per PE, in GFLOP/s
    # But only fine level runs at full PE count; use time-weighted system FLOP/s
    # Approximation: per-PE achieved / per-PE peak (same ratio at system level)
    s['Compute'] = (sys_achieved_gflops / PE_PEAK_GFLOPS) * 100

    # 5. Memory BW (system achieved / system peak)
    mem_per_v = flops_per_v / ai
    pe_bw_gbs = (mem_per_v / avg_us) * 1e-3
    s['Memory BW'] = min(100.0, (pe_bw_gbs / PE_PEAK_BW_GBS) * 100)

    return s, {
        'fine_pes': fine_pes,
        'sys_peak_tflops': sys_peak_gflops / 1e3,
        'sys_peak_mem_pbs': sys_peak_bw_gbs / 1e6,
        'sys_peak_fab_tbs': sys_peak_fab_gbs / 1e3,
    }


# ── Plotting ───────────────────────────────────────────────────────────────

def compute_per_level_spider(levels_data, fine_pes):
    """Compute 5-axis spider metrics for each level. All 0-100%.

    Axes (all relative to per-PE or allocated peaks — consistent baseline):
      1. Active PEs     — active_pes / fine_pes (what fraction of allocated PEs work)
      2. Compute Eff.   — per-PE achieved FLOP/s / 1.75 GFLOP/s peak
      3. Memory BW      — per-PE achieved BW / 28 GB/s peak
      4. Fabric BW      — per-PE achieved fabric BW / 14 GB/s peak
      5. Compute Frac.  — compute time / total SpMV time (useful work vs comm)
    """
    categories = ['Active PEs', 'Compute Eff.', 'Memory BW', 'Fabric BW', 'Compute Frac.']
    fab_peak = 4 * 4 * PE_FREQ_MHZ * 1e-3  # 14.0 GB/s

    per_level = []
    for ld in levels_data:
        vals = {
            'Active PEs':    ld['active_pes'] / fine_pes * 100,
            'Compute Eff.':  ld['compute_eff'],
            'Memory BW':     min(100, ld['mem_bw_pct']),
            'Fabric BW':     min(100, ld['fab_bw_pct']),
            'Compute Frac.': ld['compute_frac'],
        }
        per_level.append(vals)

    return categories, per_level


def make_spider_plot(categories, per_level, levels_data, size, fine_pes, output_path):
    """Multi-level spider: all levels overlaid on the same radar chart."""
    import matplotlib.cm as cm

    N = len(categories)
    angles = [n / N * 2 * pi for n in range(N)]
    angles += angles[:1]

    n_levels = len(per_level)
    colors = cm.plasma(np.linspace(0.15, 0.85, n_levels))

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)

    for i, (vals, ld, color) in enumerate(zip(per_level, levels_data, colors)):
        v = [vals[c] for c in categories] + [vals[categories[0]]]
        lw = 2.5 if i == 0 else 1.5  # fine level thicker
        alpha_fill = 0.15 if i == 0 else 0.03
        ms = 6 if i == 0 else 4
        ax.plot(angles, v, 'o-', linewidth=lw, color=color, markersize=ms,
                zorder=5 - i, markeredgecolor='black', markeredgewidth=0.5,
                label=f'L{ld["level"]} (nz={ld["nz"]}, {ld["active_pes"]:,} PEs)')
        if i <= 2:  # fill only top few levels to avoid visual noise
            ax.fill(angles, v, alpha=alpha_fill, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=11, fontweight='bold')
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'],
                       size=9, color='#666666')
    ax.set_ylim(0, 115)
    ax.grid(color='#999999', linestyle='-', linewidth=0.4)
    ax.spines['polar'].set_visible(False)

    # Legend outside
    ax.legend(loc='upper left', bbox_to_anchor=(-0.25, -0.05),
              fontsize=8, framealpha=0.9, ncol=2)

    # Info
    info = (f'{size}\u00b3  |  {fine_pes:,} PEs '
            f'({fine_pes/TOTAL_WSE3_PES*100:.0f}% of wafer)')
    ax.set_title(info, fontsize=10, pad=15, color='#444444')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved: {output_path}")


# ── Per-level table ───────────────────────────────────────────────────────

def parse_per_level_data(text):
    """Parse per-level FLOP counters, timing, and active PEs from response."""
    levels_data = []

    # FLOP counter blocks: "Level N (nz=X, active PEs: Y, stride: Z):"
    header_pat = re.compile(
        r'Level (\d+) \(nz=(\d+), active PEs:\s*([\d,]+), stride:\s*(\d+)\):\s*\n'
        r'Operation.*\n-+\n'
        r'((?:(?:FSUB|FMAC|FMUL|FADD|FNEG|FMOV_MEM|FMOV_ZERO|FMAX|FMOV32)\s+[\d,]+.*\n)*)',
        re.MULTILINE
    )

    m_iters = re.search(r'Device iterations(?:\s+to\s+converge)?\s*:\s*(\d+)', text)
    device_iters = int(m_iters.group(1)) if m_iters else 1

    # Per-level timing from first table
    main_table = re.search(
        r'Time per operation and level.*?\n(.*?)(?=\n\s*7-pt Stencil|\n\n\n)',
        text, re.DOTALL)
    level_times = {}
    level_conv_times = {}
    if main_table:
        for rm in re.finditer(
                r'\|\s+(\d+)\s+\|.*?([\d.]+)us\([^)]*\)\s*\|\s*([\d.]+)us\([^)]*\)\s*\|\s*$',
                main_table.group(1), re.MULTILINE):
            # This captures level, second-to-last col (convergence), last col (total)
            pass
        # Simpler: parse all 8-column rows
        row_pat = re.compile(
            r'\|\s*(\d+)\s*\|'
            r'\s*([\d.]+)us\([^)]*\)\s*\|'  # smooth
            r'\s*([\d.]+)us\([^)]*\)\s*\|'  # residual
            r'\s*([\d.]+)us\([^)]*\)\s*\|'  # restriction
            r'\s*([\d.]+)us\([^)]*\)\s*\|'  # interpolation
            r'\s*([\d.]+)us\([^)]*\)\s*\|'  # setup
            r'\s*([\d.]+)us\([^)]*\)\s*\|'  # convergence
            r'\s*([\d.]+)us\([^)]*\)\s*\|'  # total
        )
        for rm in row_pat.finditer(main_table.group(1)):
            lvl = int(rm.group(1))
            level_times[lvl] = float(rm.group(8))  # total
            level_conv_times[lvl] = float(rm.group(7))  # convergence

    # Stencil comm/compute per level
    stencil_section = re.search(
        r'7-pt Stencil.*?\n(.*?)(?:\n\nInterpolation|\n\n\n)', text, re.DOTALL)
    level_spmv = {}
    if stencil_section:
        spmv_row = re.compile(
            r'\|\s*(\d+)\s*\|'
            r'\s*([\d.]+)us\([^)]*\)\s*\|'  # total spmv
            r'\s*([\d.]+)us\([^)]*\)\s*\|'  # comm
            r'\s*([\d.]+)us\([^)]*\)\s*\|'  # compute
        )
        for rm in spmv_row.finditer(stencil_section.group(1)):
            lvl = int(rm.group(1))
            level_spmv[lvl] = {
                'total': float(rm.group(2)),
                'comm': float(rm.group(3)),
                'compute': float(rm.group(4)),
            }

    for match in header_pat.finditer(text):
        lvl = int(match.group(1))
        nz = int(match.group(2))
        active_pes = int(match.group(3).replace(',', ''))
        stride = int(match.group(4))

        # Count FLOPs and memory traffic
        ops_text = match.group(5)
        total_flops = 0
        total_mem_words = 0
        fab_loads = 0
        for line in ops_text.strip().split('\n'):
            parts = line.split()
            if len(parts) >= 2:
                op = parts[0].lower()
                count = int(parts[1].replace(',', ''))
                total_flops += count * FLOP_PER_OP.get(op, 0)
                total_mem_words += count * MEM_TRAFFIC.get(op, 0)
                if op == 'fmov32':
                    fab_loads += count
                    total_mem_words += count  # 1 SRAM store per fabric receive

        # Per-V-cycle values
        flops_1v = total_flops / device_iters
        mem_bytes_1v = total_mem_words * 4 / device_iters
        fab_bytes_1v = fab_loads * 4 / device_iters

        # Timing
        total_time_us = level_times.get(lvl, 0)
        conv_us = level_conv_times.get(lvl, 0)
        vcycle_us = total_time_us - conv_us  # pure V-cycle time (all iters)
        vcycle_1v_us = vcycle_us / device_iters if device_iters else 0

        # Achieved metrics
        vcycle_1v_s = vcycle_1v_us * 1e-6
        achieved_gflops = (flops_1v / vcycle_1v_s / 1e9) if vcycle_1v_s > 0 else 0
        compute_eff = achieved_gflops / PE_PEAK_GFLOPS * 100

        mem_bw_gbs = (mem_bytes_1v / vcycle_1v_us * 1e-3) if vcycle_1v_us > 0 else 0
        mem_bw_pct = mem_bw_gbs / PE_PEAK_BW_GBS * 100

        fab_bw_gbs = (fab_bytes_1v / vcycle_1v_us * 1e-3) if vcycle_1v_us > 0 else 0
        fab_bw_peak = 4 * 4 * PE_FREQ_MHZ * 1e6 / 1e9  # 14.0 GB/s
        fab_bw_pct = fab_bw_gbs / fab_bw_peak * 100

        # Compute fraction from stencil data
        spmv = level_spmv.get(lvl, {})
        compute_frac = (spmv['compute'] / spmv['total'] * 100) if spmv.get('total') else 0

        levels_data.append({
            'level': lvl, 'nz': nz, 'active_pes': active_pes, 'stride': stride,
            'flops_1v': flops_1v, 'mem_bytes_1v': mem_bytes_1v, 'fab_bytes_1v': fab_bytes_1v,
            'vcycle_1v_us': vcycle_1v_us,
            'achieved_gflops': achieved_gflops, 'compute_eff': compute_eff,
            'mem_bw_gbs': mem_bw_gbs, 'mem_bw_pct': mem_bw_pct,
            'fab_bw_gbs': fab_bw_gbs, 'fab_bw_pct': fab_bw_pct,
            'compute_frac': compute_frac,
        })

    return sorted(levels_data, key=lambda x: x['level'])


def _fmt(val, unit=''):
    """Format large numbers with K/M/T/P suffixes."""
    if val >= 1e15:
        return f'{val/1e15:.1f}P{unit}'
    elif val >= 1e12:
        return f'{val/1e12:.1f}T{unit}'
    elif val >= 1e9:
        return f'{val/1e9:.1f}G{unit}'
    elif val >= 1e6:
        return f'{val/1e6:.1f}M{unit}'
    elif val >= 1e3:
        return f'{val/1e3:.1f}K{unit}'
    else:
        return f'{val:.1f}{unit}'


def print_tables(filepath):
    """Print two tables:
    1. Resource summary — one row per metric, achieved vs peak vs %, with bottleneck note
    2. Per-level detail — per-PE metrics at each multigrid level
    """
    with open(filepath) as f:
        text = f.read()

    levels_data = parse_per_level_data(text)
    if not levels_data:
        print("No per-level data found.")
        return

    size = _fint(r'HeightxWidthxZDim\s*:\s*(\d+)', text)
    n_levels = _fint(r'Levels\s*:\s*(\d+)', text)
    fine_pes = size * size
    fab_bw_per_pe_gbs = 4 * 4 * PE_FREQ_MHZ * 1e-3  # 14.0 GB/s

    # SRAM
    sm = re.search(r'Total accounted:\s+(\d+)\s+bytes', text)
    sram_bytes = int(sm.group(1)) if sm else 0
    sram_pct = sram_bytes / SRAM_PER_PE_BYTES * 100

    # PE utilization (time-weighted)
    lt = {ld['level']: ld['vcycle_1v_us'] for ld in levels_data}
    if lt and fine_pes > 0:
        num = sum((ld['active_pes'] / fine_pes) * ld['vcycle_1v_us'] for ld in levels_data)
        den = sum(ld['vcycle_1v_us'] for ld in levels_data)
        pe_util = (num / den * 100) if den else 100
    else:
        pe_util = 0

    # Fine level (L0) metrics for summary
    l0 = levels_data[0]

    # ── Table 1: Resource Summary ──
    w = 85
    print()
    print('=' * w)
    print(f'  Resource Utilization: {size}^3, {n_levels} levels, 6/6/6')
    print('=' * w)
    print(f'  {"Metric":<25} {"Achieved":>15} {"Peak":>15} {"%":>6}  {"Bottleneck"}')
    print('-' * w)

    rows = [
        ('SRAM (per PE)',
         f'{sram_bytes/1024:.1f} KB', f'{SRAM_PER_PE_BYTES/1024:.0f} KB',
         sram_pct, 'Hard scaling limit'),
        ('Chip Area',
         f'{fine_pes:,} PEs', f'{TOTAL_WSE3_PES:,} PEs',
         fine_pes / TOTAL_WSE3_PES * 100, 'SRAM-limited'),
        ('PE Utilization',
         f'{pe_util:.0f}%', '100%',
         pe_util, 'MG hierarchy (coarse levels)'),
        ('Compute (L0)',
         f'{_fmt(l0["achieved_gflops"]*1e9*l0["active_pes"], "FLOP/s")}',
         f'{_fmt(PE_PEAK_GFLOPS*1e9*l0["active_pes"], "FLOP/s")}',
         l0['compute_eff'], 'Instruction mix (53% FMAC)'),
        ('Memory BW (L0)',
         f'{_fmt(l0["mem_bw_gbs"]*1e9*l0["active_pes"], "B/s")}',
         f'{_fmt(PE_PEAK_BW_GBS*1e9*l0["active_pes"], "B/s")}',
         l0['mem_bw_pct'], 'Compute-bound (AI=0.11)'),
        ('Fabric BW (L0)',
         f'{_fmt(l0["fab_bw_gbs"]*1e9*l0["active_pes"], "B/s")}',
         f'{_fmt(fab_bw_per_pe_gbs*1e9*l0["active_pes"], "B/s")}',
         l0['fab_bw_pct'], '7-pt stencil is mostly local'),
    ]

    for label, achieved, peak, pct, note in rows:
        print(f'  {label:<25} {achieved:>15} {peak:>15} {pct:>5.1f}%  {note}')

    print('=' * w)

    # ── Table 2: Per-Level Detail ──
    w2 = 100
    print()
    print('=' * w2)
    print(f'  Per-level per-PE metrics: {size}^3, {n_levels} levels')
    print(f'  (All BW/FLOP values are per active PE; % is vs per-PE peak)')
    print('=' * w2)
    print(f'  {"Lvl":>3} {"nz":>4} {"Active PEs":>11} {"Time(us)":>9} '
          f'{"GFLOP/s":>8} {"%pk":>5} '
          f'{"MemBW(GB/s)":>12} {"%pk":>5} '
          f'{"FabBW(GB/s)":>12} {"%pk":>5} '
          f'{"Comp%":>6}')
    print('-' * w2)

    for ld in levels_data:
        print(f'  {ld["level"]:>3} {ld["nz"]:>4} {ld["active_pes"]:>11,} {ld["vcycle_1v_us"]:>9.1f} '
              f'{ld["achieved_gflops"]:>8.3f} {ld["compute_eff"]:>4.1f}% '
              f'{ld["mem_bw_gbs"]:>12.2f} {ld["mem_bw_pct"]:>4.1f}% '
              f'{ld["fab_bw_gbs"]:>12.3f} {ld["fab_bw_pct"]:>4.1f}% '
              f'{ld["compute_frac"]:>5.1f}%')

    print('=' * w2)


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Spider plot \u2014 GMG V-cycle hardware utilization')
    parser.add_argument('--size', type=int, default=512,
                        help='Problem size (default: 512)')
    parser.add_argument('--config', default='P6_P6_B6',
                        help='Config suffix (default: P6_P6_B6)')
    parser.add_argument('--output', default=None)
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(base_dir)

    # Find response file
    pattern = os.path.join(parent_dir, 'build',
                           f'out_dir_S{args.size}x_L*_M100_{args.config}/response.txt')
    files = glob.glob(pattern)
    if not files:
        files = glob.glob(os.path.join(parent_dir,
                                       f'out_dir_S{args.size}x_L*_M100_{args.config}/response.txt'))
    if not files:
        print(f"No response file for size={args.size}, config={args.config}")
        sys.exit(1)

    resp_file = files[0]
    d = parse_response(resp_file)

    # Fallback to all_responses_roofline.txt if response is truncated
    if d.get('avg_vcycle_us') is None:
        agg = os.path.join(parent_dir, 'build', 'all_responses_roofline.txt')
        if os.path.exists(agg):
            dirname = os.path.basename(os.path.dirname(resp_file))
            with open(agg) as f:
                agg_text = f.read()
            m = re.search(
                rf'Output directory:\s*{re.escape(dirname)}\n(.*?)(?=\nOutput directory:|\Z)',
                agg_text, re.DOTALL)
            if m:
                d = parse_text(f"Output directory: {dirname}\n" + m.group(1))

    # Parse per-level data for multi-level spider
    with open(resp_file) as f:
        resp_text = f.read()
    levels_data = parse_per_level_data(resp_text)

    if not levels_data:
        print("No per-level FLOP counter data found in response.")
        sys.exit(1)

    size = d.get('size', args.size)
    fine_pes = size * size

    # Multi-level spider plot
    categories, per_level = compute_per_level_spider(levels_data, fine_pes)
    out = args.output or os.path.join(base_dir, 'spider_plot.png')
    make_spider_plot(categories, per_level, levels_data, size, fine_pes, out)

    # Tables: resource summary + per-level detail
    print_tables(resp_file)


if __name__ == '__main__':
    main()
