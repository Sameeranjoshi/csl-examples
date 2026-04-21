# Geometric Multigrid (GMG) V-Cycle Solver on Cerebras WSE-3

Device-resident Geometric Multigrid V-cycle solver for 3D Poisson equation on the Cerebras WSE-3 wafer-scale engine.

## Prerequisites

### Tooling versions (evaluated configuration)

| Component | Version |
|---|---|
| Cerebras SDK | **1.4.0** (release 2.5.0 — `cerebras-appliance==2.5.0`, `cerebras-sdk==2.5.0`) |
| SDK container image | `sdk-cbcore-202505010205-2-ef181f81.sif` (shipped with SDK 1.4.0) |
| `cslc` compiler | Bundled with SDK 1.4.0 (no separate install) |
| Python | **3.8.17** (3.8+ required) |
| Hardware | Cerebras CS-3 appliance (device runs); host-only mode needs no WSE |

The top-level `cslc` and `cs_python` entry points are thin wrappers around the
SIF image; installing the Cerebras SDK provides both.

### Installing the Cerebras SDK

Follow the vendor instructions shipped with the SDK tarball
(`Cerebras-SDK-1.4.0.tar.gz`). The SDK ships its own pinned Python environment
— activate it before running anything in this repo:

```bash
source /path/to/sdk_venv/bin/activate
pip install -r /path/to/sdk/req.txt   # installs cerebras-sdk, cerebras-appliance, and dependencies
```

### Project dependencies

The additional host-side and plotting dependencies used by this artifact are
listed in [`requirements.txt`](requirements.txt):

```bash
pip install -r requirements.txt
```

| Package | Version | Used by |
|---|---|---|
| `numpy` | 1.24.4 | host solver, plots |
| `matplotlib` | 3.7.5 | plots |
| `scipy` | 1.10.1 | host reference solver |
| `pandas` | 2.0.3 | `optimized_vs_unoptimized.py` |
| `numba` | 0.58.1 | optional; accelerates `python_gmg/gmgoscar.py` |

All versions are compatible with the SDK 1.4.0 Python environment and the
versions listed in the SDK's `req.txt` — no extra pinning conflicts.

## Quick Start

### Run on device (compile + execute)
```bash
python compile_and_run_wse3.py --only-device
```

## Problem Configuration

Edit the `configs` list in `compile_and_run_wse3.py`, comment/uncomment for specific problem size:

```python
# (size, levels, max_ite, abs_tolerance, pre_iter, post_iter, bottom_iter)
configs = [
    (4, 2, 100, 1e-5, 6, 6, 6),     # 4x4x4, 2 levels
    (128, 7, 100, 1e-5, 6, 6, 6),    # 128x128x128, 7 levels
    (512, 9, 100, 1e-5, 6, 6, 6),    # 512x512x512, 9 levels (largest)
]
```

**Parameter guide:**
- `size`: Grid dimension (size x size x size domain, size x size PEs)
- `levels`: Multigrid levels (deep = log2(size)+1, shallow = log2(size)-1)
- `max_ite`: Maximum V-cycle iterations
- `abs_tolerance`: Convergence tolerance (relative = abs * ||b||_2)
- `pre_iter / post_iter`: Jacobi smoothing iterations per level
- `bottom_iter`: Coarse-level solver iterations

**Block sizes** are auto-mapped: `{4:4, 8:8, 16:16, 32:32, 64:64, 128:128, 256:256, 512:256}`

## Output Directory Naming

Runs are written under `build/`:

```
build/out_dir_S{size}x_L{levels}_M{max_ite}_P{pre}_P{post}_B{bottom}/
```
Example: `build/out_dir_S512x_L9_M100_P6_P6_B6` = 512^3 grid, 9 levels, max 100 iterations, 6/6/6 smoothing.

Shallow variants: `build/shallow_*`.

Each directory contains:
- `response.txt` — Compile/run log with timing, convergence, and memory data
- `cs_<hash>/` — Compiled artifacts (ELF files)

## Artifact Caching

`artifact_cache.json` maps output directory names to compiled artifact paths. When `compile_and_run_wse3.py` runs, it checks this cache before compiling. 

## Reproducing Paper Figures

After configured runs have completed (outputs land in `build/out_dir_*/`):

```bash
cd plots/
bash GENERATEFIGURES.sh
```

The script runs the full pipeline:
1. Appends per-run memory usage data to each `response.txt` (from the `.tar.gz` ELF archives)
2. Aggregates `build/out_dir_*/response.txt` by configuration into `build/all_responses_*.txt`
3. Runs `plot_gmg_performance.py` on each aggregate → `out_*.txt` and per-operation timing plots (`spmv_internal.png`, `interpolation_internal.png`, `per_operation_timing_*.png`)
4. Runs `h200_vs_cs3.py` (GH200 vs WSE-3 bar chart), `memory_utilization_table.py`, `print_512_table.py`
5. Runs `v_vs_w_cycle.py` (V vs W cycle comparison)
6. Runs `roofline_analysis.py` on the 512³ 6/6/6 sample (`roofline_plot.png`)

### Running individual scripts

```bash
cd plots/

# Per-operation timing + SPMV/interpolation breakdowns
python plot_gmg_performance.py ../build/all_responses_6_6_6.txt

# GH200 vs WSE-3 bar chart (reads gpu_numbers.txt + wse_numbers.txt from cwd)
python h200_vs_cs3.py

# 512^3 comparison table
python print_512_table.py

# Memory utilization table (needs out_6_6_6.txt in cwd)
python memory_utilization_table.py out_6_6_6.txt

# V vs W cycle
python v_vs_w_cycle.py

# Roofline analysis
python roofline_analysis.py ../build/out_dir_S512x_L9_M100_P6_P6_B6/response.txt
```

### Data files
- `gpu_numbers.txt`, `wse_numbers.txt` — GH200 and WSE-3 baseline numbers consumed by `h200_vs_cs3.py` and `print_512_table.py`
- `build/all_responses_*.txt` — Concatenated device run outputs per configuration

## Directory Structure

```
csl_gmg/
├── compile_and_run_wse3.py    # Main entry point (compile + run + cache)
├── run_gmg_vcycle.py          # Device execution orchestration
├── run_gmg.py                 # Legacy host-controlled runner
├── cmd_parser.py              # Command-line argument parsing
├── util.py                    # Data layout conversions (column-major)
├── artifact_cache.json        # Build artifact cache
├── commands_vcycle_wse3.sh    # Quick test script
├── check_memory_usage.sh      # ELF memory analysis (code vs data)
├── analyze_memory.py          # Theoretical memory calculation
├── clean.sh                   # Remove build artifacts
│
├── src/                       # CSL device code
│   ├── kernel_gmg_vcycle.csl  # State machine kernel (28 states, 1103 lines)
│   ├── layout_gmg_vcycle.csl  # PE grid layout + color allocation
│   ├── blas.csl               # Linear algebra utilities
│   ├── timer_modified.csl     # Hardware timestamp management
│   └── modified_csl_lib_hops/ # Forked stencil/allreduce with hop support
│       ├── stencil_3d_7pts/   # 7-point stencil with strided communication
│       └── allreduce/         # Windowed reduction + broadcast
│
├── python_gmg/                # Reference Python solver
│   └── gmgoscar.py            # SimpleGMG class (CPU reference)
│
├── plots/                     # Analysis and visualization
│   ├── GENERATEFIGURES.sh     # Reproduce all paper figures (full pipeline)
│   ├── plot_gmg_performance.py # Per-operation timing analysis
│   ├── h200_vs_cs3.py         # GH200 vs WSE-3 bar chart
│   ├── memory_utilization_table.py # Per-level memory usage table
│   ├── print_512_table.py     # 512^3 comparison table
│   ├── v_vs_w_cycle.py        # V vs W cycle comparison
│   ├── roofline_analysis.py   # Roofline plot
│   ├── gpu_numbers.txt        # GH200 baseline numbers
│   └── wse_numbers.txt        # WSE-3 baseline numbers
│
├── docs/                      # Artifact evaluation reproduction guide
│   └── REPRODUCE.md
│
├── paper/                     # ICS GLOW 2026 paper
│   ├── rebuttal.md            # Rebuttal responses
│   └── ICS_GLOW_2026/         # LaTeX source + figures
│
└── build/                     # Run outputs (gitignored)
    ├── out_dir_S*x_*/         # Per-config device run outputs
    ├── shallow_*/             # Shallow V-cycle outputs
    └── all_responses_*.txt    # Aggregated response files
```

## Artifact Evaluation

See [`docs/REPRODUCE.md`](docs/REPRODUCE.md) for a consolidated reproduction guide.
