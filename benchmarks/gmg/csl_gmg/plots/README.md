# GMG V-cycle plotting pipeline

Regenerate every figure used in the paper/report with one command:

```bash
cd plots/
./GENERATEFIGURES.sh
```

The script runs in 5 stages (see `GENERATEFIGURES.sh` for details). It tolerates missing inputs and prints a `SKIP` line for anything it can't run.

## Prerequisites

- Hardware runs completed in `../out_dir_S*x*_P*_P*_B*/response.txt`
- Python 3 with `matplotlib` and `numpy`

## Scripts

| Script | Input | Output | What it shows |
|---|---|---|---|
| `plot_gmg_performance.py` | `all_responses_*.txt` | `out_*.txt` + `spmv_internal.png`, `interpolation_internal.png`, `per_operation_timing_512x512x512.png` | Parses response.txt, prints per-size summary tables, plots SpMV (comm/compute) and interpolation breakdown per level |
| `time_to_solution.py` | aggregated response files | `time_to_solution.png`, `iterations_comparison.png` | WSE-3 TTS vs GH200 HPGMG across grid sizes and configs |
| `spider_plot.py` | `../out_dir_S256*` response | `spider_plot.png` | Radar chart of 256³ metrics (compute/mem/fabric efficiency, PE utilization, etc.) |
| `wafer_utilization.py` | aggregated response files | `wafer_utilization.png` | Per-level active PE count and fabric footprint |
| `fixed_tolerance_analysis.py` | aggregated response files | `fixed_tolerance_convergence.png` | Iterations to cross a fixed absolute tolerance (1e-5) |
| `plot_convergence.py` | `all_responses_*.txt` | `convergence_by_size.png`, `convergence_by_config.png`, `convergence_key_sizes.png` | rho-history curves per iteration for each config |
| `correctness_check.py` | `../out_dir_S*` response | stdout (or `correctness_check.png` with `--run-host`) | Verifies device rho matches host Python reference |
| `roofline_analysis.py` | single response.txt | `roofline_plot.png` (+ stdout tables) | 4-panel roofline (per-PE, per-level active, full grid, combined) |
| `optimized_vs_unoptimized.py` | `out_6_6_6.txt` + `out_6_6_6_unoptimized.txt` | `metrics_comparison.png` | Before/after metric comparison |
| `h200_vs_cs3.py` | `h200_vs_cs3_feb7.csv` | `hpgmg_speedup_barplot.pdf` | WSE-3 vs GH200 speedup bar chart |

## Standard timer methodology (matches HPGMG comparison)

All plots use this single, consistent V-cycle time definition. This is emitted by
`run_gmg_vcycle.py` in the summary block at the end of `response.txt`:

```
Total solver wall time (all N V-cycles, inc. conv check per V-cycle): WALL us
Convergence diagnostic time (total, N checks, only at L0):           CONV us
Pure operators time (wall - convergence, all N V-cycles):            PURE us
Average V-cycle time (pure operators / N iters, no conv):            AVG  us
```

- `WALL` — whole solver wall time, includes one convergence check per V-cycle
- `CONV` — total convergence-check time across N iterations (accumulates at L0 only)
- `PURE = WALL - CONV` — pure V-cycle operators, across all iterations
- `AVG = PURE / N` — **the number plots use** as "per-V-cycle" time

Convergence is excluded because HPGMG reports the same (operators-only V-cycle time).

The older labels `"1st V-cycle time (measured)"` and `"1st V-cycle time (sum of
per-level timers, directly measured)"` were misleading — they reported the sum
of per-level accumulated timers (i.e., TOTAL across all iterations, not the first
V-cycle). The plot scripts still parse them for backward compatibility, but new
runs use `"Avg V-cycle time (no conv)"`.

## File conventions

- `out_dir_S{size}x_L{levels}_M{max_iter}_P{pre}_P{post}_B{bottom}/response.txt` — one raw hardware run per problem size+config
- `all_responses_{pre}_{post}_{bottom}.txt` — concatenated responses for one config, sorted by size
- `out_{pre}_{post}_{bottom}.txt` — parsed summary (stdout of `plot_gmg_performance.py`) for one config; input to comparison scripts

## Handling output-format changes

`plot_gmg_performance.py` supports legacy and current response.txt formats:

- SpMV table header: three variants (old, mid, current) accepted
- V-cycle time label: multiple patterns accepted (`Wall time per V-cycle`, `1-V cycle time(Average)`, `1st V-cycle time (measured)`, `1st V-cycle time (sum of per-level timers...)`)
- Interpolation micro-benchmark header: case-insensitive match

If a new response.txt format appears and parsing stops working, add the new header/label to the parser's lookup tables (search for `SPMV_HEADER_*` and `INTERP_HEADERS` in `plot_gmg_performance.py`, and the regex alternation in `parse_vcycle_times` / `parse_configuration_summary`).
