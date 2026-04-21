# Appendix: Artifact Description (AD)

Paper: **GLOW: Accelerating Geometric Multigrid Levels on the Cerebras Wafer-Scale Engine**

---

## Part 1 — Overview of Contributions and Artifacts

### 1.A Paper's Main Contributions

- **C1.** The first implementation of a multigrid algorithm on Cerebras WS-3.
- **C2.** A performance model of communication costs, along with a forwarding communication optimization, using the strided distributed solution which we proposed.
- **C3.** Extensive per-operator and per-level performance measurements on WSE-3 along with roofline measurements. 
- **C4.** Comparison of GLOW against HPGMG on an NVIDIA GH200 (up to 25× single V-cycle speedup).
- **C5.** Design lessons regarding GMG algorithm configuration (V- vs. W-/shallow-cycle, code/data tradeoffs) for tiled architectures with limited per-tile memory.

### 1.B Computational Artifacts

All artifacts will be provided at the following DOI:
A1: (`https://doi.org/XX.YYYY/zenodo.NNNNNN`)
A2: (`hpgmg`)


| Artifact |  Contributions supported | Paper elements reproduced |
|---|---|---|
| **A1** | C1, C2, C3, C4, C5 | Figures 5–9; Tables 3, 4, 5 |
| **A2** | C4 | Provides the GH200 baseline data for Figure 5 and Table 4 |

---

## Part 2 — Artifact Identification

### Artifact A1 — GLOW

#### A1.A Relation to contributions

A1 is the GLOW implementation that realizes contributions C1–C5. It contains
the CSL device kernel, the host orchestration, the strided-distributed PE
layout, the forwarding-optimized stencil/allreduce libraries (C2), and the
plotting scripts that regenerate every figure and table attributed to WSE-3
measurements in the paper (C3, C4).

#### A1.B Expected Results

After running A1 the reviewer should see:

- Per-configuration response files (`build/out_dir_*/response.txt`) containing
  compile commands, per-V-cycle wall-clock timings, convergence residuals, and
  per-PE memory usage.
- A set of PDF/PNG figures in `plots/`:
  - `hpgmg_speedup_barplot.pdf` — Single V-cycle speedup of GLOW (WSE-3)
    over HPGMG (GH200) across grid sizes and V-cycle configurations
    (substantiates C3).
  - `per_operation_timing_512x512x512.png` — Per-operator wall time at each
    multigrid level for the 512³ problem (substantiates C3).
  - `spmv_internal.png`, `interpolation_internal.png` — Compute-vs-
    communication micro-benchmarks for the strided stencil and
    interpolation operators (substantiates C2).
  - `roofline_plot.png` — Per-PE and system-level roofline for the 512³
    6/6/6 run (substantiates C3, C4).
- Tables printed to stdout / captured in `GENERATEFIGURES.log`:
  - 512³ platform comparison table (TTS, 1-cycle time, #cycles) —
    `print_512_table.py`.
  - Memory utilization per level — `memory_utilization_table.py`.
  - V- vs. W-cycle comparison — `v_vs_w_cycle.py`.

These outcomes substantiate C3 (the 25× headline speedup and the per-operator
breakdown) and, together with the 512³ memory utilization table, C4 (the
code/data tradeoff that bounds the largest feasible problem).

#### A1.C Expected Reproduction Time

| Step | Estimate |
|---|---|
| Artifact Setup (SDK install + venv + `pip install -r requirements.txt`) | **30–45 min** |
| Artifact Execution — full sweep (compile + run all configured sizes 4³…512³) on CS-3 | **45–90 min** (compilation dominates; with `artifact_cache.json` re-runs take **<10 min**) |
| Artifact Analysis — `bash plots/GENERATEFIGURES.sh` (aggregation + plots + tables) | **2–5 min** |
| **Total (cold)** | **~80–140 min** |

Host-only mode (no CS-3 access) runs the reference solver in Python and takes
~10–30 min for sizes up to 64³; it validates correctness but cannot reproduce
the speedup claims.

#### A1.D Artifact Setup

**Hardware.**
- **Primary:** access to a Cerebras CS-3 system (WSE-3, 762×1172 tile fabric,
  ~40 GB on-wafer SRAM). The artifact targets a single wafer; no multi-wafer
  configuration is needed.
- **For baseline comparison (A2):** an NVIDIA GH200 Grace-Hopper node (or
  equivalent H100/H200) with ≥80 GB HBM and CUDA-capable PCIe/NVLink host.
- **Host CPU** for compiling/driving the appliance: any x86_64 Linux host
  that can run the Cerebras SDK container image.

**Software.**

| Package | Version | Source |
|---|---|---|
| Cerebras SDK | 1.4.0 (release 2.5.0: `cerebras-appliance==2.5.0`, `cerebras-sdk==2.5.0`) | Cerebras, Inc. (vendor distribution) |
| SDK container (SIF) | `sdk-cbcore-202505010205-2-ef181f81.sif` | Shipped with SDK 1.4.0 |
| `cslc` | Bundled with SDK 1.4.0 | Vendor |
| Python | 3.8.17 (3.8+ required) | https://www.python.org/ |
| NumPy | 1.24.4 | https://pypi.org/project/numpy/ |
| SciPy | 1.10.1 | https://pypi.org/project/scipy/ |
| Matplotlib | 3.7.5 | https://pypi.org/project/matplotlib/ |
| Pandas | 2.0.3 | https://pypi.org/project/pandas/ |
| Numba (optional) | 0.58.1 | https://pypi.org/project/numba/ |
| CUDA toolkit (for A2) | ≥ 12.x | https://developer.nvidia.com/cuda-toolkit |
| GCC + OpenMP + MPI (`mpicc`, for A2) | system defaults | — |

**Datasets / Inputs.**
GLOW solves the 3D Poisson equation `Ax = b` with a 7-point stencil
(`α = −6.0, β = 1.0`) and a synthetic right-hand side generated at runtime by
the host driver (`run_gmg_vcycle.py`). No external dataset is required.
Problem sizes are specified programmatically in the `configs` list of
`compile_and_run_wse3.py`:

```python
# (size, levels, max_ite, abs_tolerance, pre_iter, post_iter, bottom_iter)
configs = [
    (4,   2, 100, 1e-5, 6, 6, 6),
    (8,   3, 100, 1e-5, 6, 6, 6),
    (16,  4, 100, 1e-5, 6, 6, 6),
    (32,  5, 100, 1e-5, 6, 6, 6),
    (64,  6, 100, 1e-5, 6, 6, 6),
    (128, 7, 100, 1e-5, 6, 6, 6),
    (256, 8, 100, 1e-5, 6, 6, 6),
    (512, 9, 100, 1e-5, 6, 6, 6),
]
```

**Installation and Deployment.**

```bash
# 1. Activate the Cerebras SDK's Python venv and install SDK packages
source /path/to/sdk_venv/bin/activate
pip install -r /path/to/sdk/req.txt        # cerebras-sdk==2.5.0, cerebras-appliance==2.5.0, …

# 2. Install GLOW's host-side and plotting dependencies
cd benchmarks/gmg/csl_gmg_with_conv
pip install -r requirements.txt
```

Compilation uses `SdkCompiler` (no manual `cslc` invocation). The compile
flags, assembled automatically by `compile_and_run_wse3.py:355-363`, are:

```
--arch=wse3  --fabric-dims=762,1172  --fabric-offsets=4,1
--params=width:N,height:N,MAX_ZDIM:N,LEVELS:L
--params=BLOCK_SIZE:B  --memcpy  --channels=C
--width-west-buf=0  --width-east-buf=0  -o out_vcycle{N}
--llvm-option=--inline-threshold=256  --llvm-option=--unroll-threshold=256
```
where `BLOCK_SIZE` is mapped from grid size `N` via
`{4:4, 8:8, 16:16, 32:32, 64:64, 128:128, 256:256, 512:256}` and
`C = min(N, 16)`. See `docs/REPRODUCE.md` for the per-flag explanation.

#### A1.E Artifact Evaluation — Experiment Workflow

The workflow decomposes into four tasks with a linear dependency chain
`T1 → T2 → T3 → T4`:

- **T1 (Configure).** Edit `configs` in `compile_and_run_wse3.py` to enable
  the desired `(size, levels, …)` problems. For the full paper sweep, all
  eight entries from 4³ to 512³ are used.
- **T2 (Compile).** `SdkCompiler` produces one WSE-3 ELF artifact per
  `(size, levels)` pair. Successful compiles are cached in
  `artifact_cache.json`, so re-runs skip compilation.
- **T3 (Execute on CS-3).** `SdkLauncher` stages the host driver
  (`run_gmg_vcycle.py`) and launches the V-cycle solver. Each run writes a
  `build/out_dir_S{size}x_L{levels}_M{max_ite}_P{pre}_P{post}_B{bottom}/response.txt`
  capturing the compile command, per-operator 48-bit hardware timestamps
  (converted to μs via the 850 MHz WSE clock), per-V-cycle residuals, and
  per-PE memory usage (code vs. data).
- **T4 (Analyze & Plot).** `bash plots/GENERATEFIGURES.sh` appends ELF
  memory-usage data to every `response.txt`, aggregates responses into
  `build/all_responses_*.txt`, then invokes the plotting scripts
  (`plot_gmg_performance.py`, `h200_vs_cs3.py`,
  `memory_utilization_table.py`, `print_512_table.py`, `v_vs_w_cycle.py`,
  `roofline_analysis.py`).

Dependency graph:

```
T1 (edit configs)
      │
      ▼
T2 (compile) ──►  build/out_dir_*/*.tar.gz        (cached via artifact_cache.json)
      │
      ▼
T3 (execute on CS-3) ──►  build/out_dir_*/response.txt
      │
      ▼
T4 (analyze)  ──►  plots/{*.png,*.pdf}, stdout tables, GENERATEFIGURES.log
```

**Experimental parameters.**

| Parameter | Value | Rationale |
|---|---|---|
| Grid sizes | 4³, 8³, 16³, 32³, 64³, 128³, 256³, 512³ | Weak scaling — 1 PE per cell on the finest level; 512³ is the largest that fits per-PE SRAM (98% utilization) |
| Multigrid levels `L` | `log2(size) + 1` (deep); `log2(size) − 1` (shallow) | Deep cycles recurse to a 2×2 coarsest grid; shallow variants explored for C4 |
| Pre/post/bottom smoothing | 6/6/6 (primary), 4/4/6, 4/4/100, 6/6/100 | Varying smoother count exercises the compute/communication tradeoff at each level |
| Smoother | Weighted Jacobi, ω = 2/3 | Standard GMG choice, matches HPGMG baseline |
| Stencil | 7-point, matrix-free, α = −6.0, β = 1.0 | 3D Poisson; identical discretization to HPGMG |
| Convergence | abs tol = 1e-5, max iter = 100 | Same tolerance applied to both platforms |
| Precision | float32 | Matches the GH200 HPGMG build (`H100build_fp32`) |
| Channels `C` | 16 when `N > 16`, else `N` | DMA routing budget; saturates above 16×16 |
| Repetitions | 1 run per configuration | Hardware timestamps are deterministic on WSE-3; multi-run variance is reported as <1% in the paper |

The HPGMG baseline (A2) is built with
`H100build_fp32`/`H200build_fp32` (CUDA_ARCH `sm_90`), `MAX_SOLVES=10`,
`USE_DIRICHLET_BC`, and `fv-smoother=jacobi` (see `hpgmg/build.sh`). Each
problem size is driven by `./bin/hpgmg-fv <log2(size)> 1` (see `hpgmg/run.sh`).

#### A1.F Artifact Analysis

Raw timestamps written to each `response.txt` are converted to final paper
figures/tables by `bash plots/GENERATEFIGURES.sh`, which runs this pipeline:

1. **Memory append** — for each `build/out_dir_S*/response.txt`, extract the
   ELF from the cached `.tar.gz`, run `check_memory_usage.sh` to parse
   `FUNC`/`OBJ` symbols, and append the per-PE code/data usage to
   `response.txt`.
2. **Aggregation** — concatenate per-config `response.txt` files into
   `build/all_responses_{6_6_6,4_4_6,4_4_100,6_6_100,6_6_6_shallow,6_6_6_unoptimized}.txt`,
   sorted by grid size.
3. **Per-operator timing** — `plot_gmg_performance.py` parses the 13+
   per-operator timing arrays (smooth, residual, restrict, interp, SPMV
   compute, SPMV communication, interp expand_z, bcast, …), converts
   48-bit cycle counts to μs using the 850 MHz WSE clock, and emits
   `per_operation_timing_*.png`, `spmv_internal.png`,
   `interpolation_internal.png`, and structured `out_*.txt` summaries.
4. **Cross-platform comparison** — `h200_vs_cs3.py` reads the aggregated
   `wse_numbers.txt` (from step 3) and `gpu_numbers.txt` (from the A2 HPGMG
   runs) and produces `hpgmg_speedup_barplot.pdf`. `print_512_table.py`
   emits the 512³ TTS / 1-cycle / #cycles table.
5. **Memory table** — `memory_utilization_table.py` converts the
   memory-append output from step 1 into the per-level code/data/SRAM
   utilization table.
6. **Cycle comparison** — `v_vs_w_cycle.py` contrasts V- vs. W-cycle
   timings, substantiating the claims in the "On V, W and F cycles"
   section (C4).
7. **Roofline** — `roofline_analysis.py` takes a single 512³ 6/6/6
   response, counts DSD operations per level, computes per-PE and
   system-level FLOPs and memory/fabric traffic, and writes
   `roofline_plot.png`/`.pdf`.

All intermediate outputs are deterministic; re-running the pipeline on the
same `response.txt` files reproduces every figure and table bit-for-bit.

---

### Artifact A2 — HPGMG-FV (CUDA baseline on GH200)

#### A2.A Relation to contributions

A2 is the unmodified HPGMG-FV (NVIDIA fork, CUDA + Unified Memory) used as
the GPU baseline for C3. It is not a novel contribution of the paper; it is
included so the speedup numbers in the cross-platform bar chart and the 512³
comparison table are independently reproducible.

#### A2.B Expected Results

After building HPGMG and running `hpgmg/run.sh`, the reviewer obtains
`results_H200_fp32.txt` (or the equivalent for A100/H100), containing the
per-problem-size number of V-cycles to convergence and the per-cycle wall
time on the GPU. These numbers populate `plots/gpu_numbers.txt` and are
consumed by A1's analysis pipeline (`h200_vs_cs3.py`, `print_512_table.py`)
to compute the reported speedups.

#### A2.C Expected Reproduction Time

| Step | Estimate |
|---|---|
| Artifact Setup (CUDA toolkit, MPI, `configure` + `make`) | **10–20 min** |
| Artifact Execution (`run.sh` sweeping 2²…2⁹ grids) | **5–15 min** |
| Artifact Analysis (parsing `results_*.txt` into `gpu_numbers.txt`) | **<2 min** |
| **Total** | **~20–40 min** |

#### A2.D Artifact Setup

**Hardware.** NVIDIA GH200 (primary, matches the paper); H100 or H200
acceptable as alternatives. ≥ 80 GB HBM is required for 512³ fp32.

**Software.** CUDA toolkit ≥ 12.x, MPI (`mpicc`), OpenMP, GCC. HPGMG-FV
sources at `hpgmg/` (NVIDIA fork, commit bundled with the artifact). See
https://github.com/NVIDIA/hpgmg-cuda for the upstream repository.

**Datasets / Inputs.** None external; HPGMG generates its Poisson test
problem internally, controlled by the `log2(size)` argument to `hpgmg-fv`.

**Installation and Deployment.**

```bash
cd hpgmg
bash build.sh            # produces ./H200_fp32/bin/hpgmg-fv
bash run.sh              # sweeps 2^2..2^9 grids, writes results_H200_fp32.txt
```

`build.sh` pins `-gencode arch=compute_90,code=sm_90` for Hopper. For
A100/H100 builds see `A100build_fp32/` and `H100build_fp32/`. The build
enables `USE_DIRICHLET_BC`, `MAX_SOLVES=10`, and `fv-smoother=jacobi`
(matching A1's discretization).

#### A2.E Artifact Evaluation — Experiment Workflow

Linear pipeline `T1 → T2 → T3`:

- **T1 (Build).** `build.sh` invokes `./configure` with the pinned OPT flags
  and `make -j` to produce `H200_fp32/bin/hpgmg-fv`.
- **T2 (Run).** `run.sh` sweeps `./bin/hpgmg-fv <log2(size)> 1` for
  `log2(size)` in 2..9 and appends output to `results_H200_fp32.txt`.
- **T3 (Extract).** `parse_results.py` (or manual tabulation) converts the
  text dump into the two-column `gpu_numbers.txt` format (size, per-cycle
  time) consumed by A1's plotting pipeline.

Parameters match A1: fp32 precision, Dirichlet BCs, V-cycle, Jacobi smoother,
single GPU (`mpirun -np 1`), `MAX_SOLVES=10`, `OMP_NUM_THREADS=4`. One run
per problem size; HPGMG reports its own per-solve timing so no external
repetition is needed.

#### A2.F Artifact Analysis

`parse_results.py` scans `results_*.txt`, extracts the reported
time-per-solve and number-of-V-cycles per problem size, and emits the
`(size, per_cycle_time_s, n_cycles)` rows used by
`benchmarks/gmg/csl_gmg_with_conv/plots/gpu_numbers.txt`. No further
processing is required — the data flows directly into the A1 plotting
pipeline described in A1.F, which produces the cross-platform bar chart and
the 512³ comparison table.
