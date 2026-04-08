# Geometric Multigrid (GMG) V-Cycle Solver on Cerebras WSE-3

Device-resident Geometric Multigrid V-cycle solver for 3D Poisson equation on the Cerebras WSE-3 wafer-scale engine. Described in the ICS GLOW 2026 paper.

## Prerequisites

- Cerebras SDK (provides `cslc` compiler, `cs_python`, `SdkCompiler`, `SdkLauncher`)
- Python 3.8+ with `numpy`, `matplotlib`
- Access to a Cerebras CS-3 cluster (for device runs)
- Optional: `numba` (speeds up host reference solver)

## Quick Start

### Run on device (compile + execute)
```bash
python compile_and_run_wse3.py --only-device
```

### Run host reference solver only (no WSE-3 needed)
```bash
python compile_and_run_wse3.py --only-host
```

### Run both and compare
```bash
python compile_and_run_wse3.py --host-and-device
```

### Quick test (small problem)
```bash
bash commands_vcycle_wse3.sh
```

## Problem Configuration

Edit the `configs` list in `compile_and_run_wse3.py`:

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

```
out_dir_S{size}x_L{levels}_M{max_ite}_P{pre}_P{post}_B{bottom}
```
Example: `out_dir_S512x_L9_M100_P6_P6_B6` = 512^3 grid, 9 levels, max 100 iterations, 6/6/6 smoothing.

Shallow variants: `shallow_out_dir_S{size}x_...`
Unoptimized variants: `out_dir_S{size}x_..._unoptimized`

Each directory contains:
- `response.txt` — Compile/run log with timing, convergence, and memory data
- `cs_<hash>/` — Compiled artifacts (ELF files)

## Artifact Caching

`artifact_cache.json` maps output directory names to compiled artifact paths. When `compile_and_run_wse3.py` runs, it checks this cache before compiling. Delete `artifact_cache.json` to force recompilation.

## Reproducing Paper Figures

### Step 1: Collect response data (after all runs complete)
```bash
cd plots/
bash GENERATEFIGURES.sh
```

This script:
1. Concatenates `response.txt` files by configuration into `all_responses_*.txt`
2. Runs `plot_gmg_performance.py` on each to generate per-operation timing plots
3. Runs `optimized_vs_unoptimized.py` for the optimization comparison figure
4. Runs `h200_vs_cs3.py` for the speedup bar chart

### Step 2: Individual figures

```bash
cd plots/

# Per-operation timing + SPMV/interpolation breakdowns (3 PNGs)
python plot_gmg_performance.py ../all_responses_6_6_6.txt

# Optimization comparison (metrics_comparison.png)
python optimized_vs_unoptimized.py out_6_6_6.txt out_6_6_6_unoptimized.txt

# GH200 vs WSE-3 speedup bar chart (hpgmg_speedup_barplot.pdf)
python h200_vs_cs3.py h200_vs_cs3_feb7.csv

# Time-to-solution analysis (time_to_solution.png, iterations_comparison.png)
python time_to_solution.py

# Convergence plots (convergence_by_size.png, convergence_key_sizes.png)
python plot_convergence.py

# Wafer utilization analysis (wafer_utilization.png)
python wafer_utilization.py

# Correctness verification (correctness_check.png)
python correctness_check.py --run-host --sizes 4,8,16,32,64
```

### Data files
- `h200_vs_cs3_feb7.csv` — GH200 vs WSE-3 comparison data (iterations + per-V-cycle times)
- `all_responses_*.txt` — Concatenated device run outputs per configuration

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
│   ├── GENERATEFIGURES.sh     # Reproduce all paper figures
│   ├── plot_gmg_performance.py # Per-operation timing analysis
│   ├── h200_vs_cs3.py         # GH200 speedup comparison
│   ├── optimized_vs_unoptimized.py # Optimization impact
│   ├── time_to_solution.py    # TTS analysis
│   ├── plot_convergence.py    # Convergence plots
│   ├── wafer_utilization.py   # PE utilization + memory analysis
│   ├── correctness_check.py   # Host vs device verification
│   └── h200_vs_cs3_feb7.csv   # Baseline comparison data
│
├── docs/                      # Documentation
│   ├── README_STATE_MACHINE.md
│   ├── README_VCYCLE.md
│   ├── VCYCLE_STATE_MACHINE_SUMMARY.md
│   ├── TIMING_GUIDE.md
│   └── QUICK_TIMING_REFERENCE.md
│
├── paper/                     # ICS GLOW 2026 paper
│   ├── rebuttal.md            # Rebuttal responses
│   └── ICS_GLOW_2026/         # LaTeX source + figures
│
├── out_dir_S*x_*/             # Device run outputs (gitignored)
├── shallow_out_dir_S*x_*/     # Shallow V-cycle outputs
└── all_responses_*.txt        # Aggregated response files
```

## Key Implementation Details

- **State machine**: 28-state callback-driven V-cycle executes entirely on-device
- **Pencil decomposition**: Full z-dimension stored per PE (all z-accesses local)
- **Strided active PEs**: At level L, only PEs at stride 2^L are active
- **Memory**: Code constant ~22.8 KB; data follows geometric series nz(2 - 1/2^(L-1))
- **Timing**: 48-bit hardware TSC at 850 MHz, 13+ timing arrays per operation per level
- **Convergence**: |rho|_inf = max|f - Au|, checked after each V-cycle on host
