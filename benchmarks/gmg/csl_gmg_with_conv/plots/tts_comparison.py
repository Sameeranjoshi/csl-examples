#!/usr/bin/env python3
"""
Time per V-cycle comparison: GH200 vs WSE-3 across all problem sizes.

Reads h200_vs_cs3_april6.csv and prints a table with problem size,
levels, active PEs, total grid points, and per-V-cycle times.

Usage:
  python tts_comparison.py
"""

import os
import csv
import math

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(SCRIPT_DIR, 'h200_vs_cs3_april6.csv')

# Column indices (0-based) from the CSV — "1 V cycle" columns
CONFIGS = {
    '6/6/100': {'h200_1v': 3,  'wse3_1v': 16},
    '4/4/6':   {'h200_1v': 6,  'wse3_1v': 18},
    '6/6/6':   {'h200_1v': 9,  'wse3_1v': 20},
    '4/4/100': {'h200_1v': 12, 'wse3_1v': 22},
}

DATA_ROWS = range(12, 20)  # CSV rows 13-20 (0-indexed: 12-19)


def parse_csv():
    """Parse CSV and return {config: {size: (h200_1v_sec, wse3_1v_sec)}}."""
    with open(CSV_PATH) as f:
        rows = list(csv.reader(f))

    results = {cfg: {} for cfg in CONFIGS}

    for row_idx in DATA_ROWS:
        row = rows[row_idx]
        label = row[0].strip().replace('xx', 'x')  # fix "64xx64"
        size = int(label.split('x')[0])

        for cfg, cols in CONFIGS.items():
            h200_val = row[cols['h200_1v']].strip()
            wse3_val = row[cols['wse3_1v']].strip()

            h200_1v = float(h200_val) if h200_val else None
            wse3_1v = float(wse3_val) if wse3_val else None

            if h200_1v is not None or wse3_1v is not None:
                results[cfg][size] = (h200_1v, wse3_1v)

    return results


def print_table(results):
    cfg = '6/6/6'
    data = results[cfg]

    print(f"\nPer-V-cycle time ({cfg}):")
    print(f"{'Size':>6}  {'Levels':>6}  {'Active PEs':>12}  {'Total Points':>14}  "
          f"{'GH200 (ms)':>12}  {'WSE-3 (ms)':>12}  {'Speedup':>8}")
    print('-' * 80)

    for s in sorted(data.keys()):
        levels = int(math.log2(s))
        active_pes = s * s
        total_pts = s ** 3
        h, w = data[s]
        h_str = f'{h*1e3:.4f}' if h else '-'
        w_str = f'{w*1e3:.4f}' if w else '-'
        sp_str = f'{h/w:.1f}x' if h and w else '-'
        print(f'{s:>6}  {levels:>6}  {active_pes:>12,}  {total_pts:>14,}  '
              f'{h_str:>12}  {w_str:>12}  {sp_str:>8}')


def main():
    results = parse_csv()
    print_table(results)


if __name__ == '__main__':
    main()
