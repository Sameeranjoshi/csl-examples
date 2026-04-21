# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

A benchmark suite for Cerebras Wafer-Scale Engine (WSE2/WSE3) hardware. Benchmarks range from simple kernels (bandwidth-test, mandelbrot) to complex numerical solvers (geometric multigrid, conjugate gradient, BiCGSTAB). The most actively developed component is the GMG (Geometric Multigrid) solver in `gmg/`.

## Languages

- **CSL** (Cerebras Software Language): Device-side kernels (`layout.csl`, `kernel.csl`, `pe.csl`)
- **Python**: Host-side orchestration, data transfer, result collection (`run.py`)
- **Bash**: Build/run automation (`commands_wse2.sh`, `commands_wse3.sh`)

## Build and Run

There is no unified build system. Each benchmark is compiled and run independently via shell scripts.

### Compile a benchmark (CSL → ELF)
```bash
cslc ./src/layout.csl --arch wse3 --fabric-dims=12,7 --fabric-offsets=4,1 \
  --params=width:5,height:5,MAX_ZDIM:5 -o=out --memcpy --channels=1
```

### Run a benchmark
```bash
cs_python ./run.py -m=5 -n=5 -k=5 --latestlink out --channels=1 --run-only
```

### Typical workflow: run a benchmark's shell script
```bash
cd conjugate-gradient && bash commands_wse3.sh
```

### GMG (most complex benchmark)
GMG uses Python-driven compilation with artifact caching:
```bash
cd gmg/csl_gmg && python compile_and_run_wse3.py
```
It caches compiled artifacts in `artifact_cache.json` to skip recompilation.

## Architecture

### Benchmark structure (typical)
```
benchmark-name/
├── src/layout.csl       # Fabric topology, PE placement, module imports
├── src/kernel.csl       # Device computation kernel
├── run.py               # Host orchestration (memcpy H2D/D2H, launch, verify)
├── commands_wse2.sh     # Compile+run for WSE2
└── commands_wse3.sh     # Compile+run for WSE3
```

### Host execution pattern (run.py)
All run.py files follow the same flow: parse args → `SdkRuntime.load()` → `run()` → `memcpy_h2d()` → `launch("kernel_fn")` → `memcpy_d2h()` → `stop()`.

### Shared libraries (`csl-libs/`)
- **allreduce/**: Row/column reduction and broadcast across PEs
- **stencil_3d_7pts/**: Reusable 7-point stencil SpMV with north/south/east/west neighbor communication

These are imported by `layout.csl` files in benchmarks like conjugate-gradient, preconditioned-conjugate-gradient, and 7pt-stencil-spmv.

### CSL compilation model
- `layout.csl` defines the 2D PE grid, allocates colors (communication channels) and entrypoints, then assigns `kernel.csl` to each tile
- Colors are passed as `--params=C0_ID:0,C1_ID:1,...` to avoid hardcoding
- `--memcpy` flag enables host↔device data transfer infrastructure
- WSE2 vs WSE3 differ in queue management (WSE3 uses `@get_input_queue`/`@get_output_queue`)

### GMG structure (`gmg/`)
The GMG solver is decomposed into standalone kernel benchmarks:
- `kernel_interpolation/`, `kernel_restriction/`, `kernel_smooth/`, `kernel_residual/`, `kernel_spmv/`, `kernel_solver/`
- `csl_gmg/` is the full V-cycle solver integrating all kernels
- Each kernel subdirectory has its own `docs/`, `python_gmg/`, and `src/`

### Timing
`timing_library/python_libs/my_timeit.py` provides `TimingCalculator` for converting 48-bit hardware timestamps to microseconds. WSE clock runs at 850 MHz: `time(us) = (cycles / 0.85) * 1e-3`.

## GMG V-Cycle Solver (`gmg/csl_gmg/`)

This is the primary research artifact — a device-resident Geometric Multigrid V-cycle solver described in the ICS GLOW 2026 paper. It achieves up to 25-26× speedup over HPGMG on NVIDIA GH200.

### Algorithm
Solves sparse linear systems **Ax = b** using a 7-point stencil (matrix-free) with a V-cycle that recurses through grid levels:
1. **Down-cycle**: Pre-smooth (weighted Jacobi, ω=2/3) → Residual (r = f - Au) → Restriction (average 8 fine → 1 coarse)
2. **Bottom solve**: Iterative Jacobi at coarsest 2×2 grid
3. **Up-cycle**: Interpolation (1 coarse → 8 fine via broadcast) → Error correction → Post-smooth

### Hardware mapping (Strided Distributed)
- **Pencil decomposition**: Entire z-dimension stored per PE (all z-accesses local, no communication)
- **Strided active PEs**: At level L, only PEs at stride 2^L are active — coarser levels use fewer, more distant PEs
- **Memory**: Per-PE allocation follows geometric series: nz(2 - 1/2^(L-1)). Code section uses ~47% of 48KB SRAM; remaining ~25KB for data across all levels
- Grid sizes from 4×4×4 to 512×512×512 supported; block sizes: `{4:4, 8:8, 16:16, 32:32, 64:64, 128:128, 256:256, 512:256}`

### Key source files
- `compile_and_run_wse3.py` — Entry point: compiles CSL, launches jobs, manages artifact cache, has `--only-host`, `--only-device`, `--host-and-device` modes
- `run_gmg_vcycle.py` — Host orchestration: copies data to device, launches V-cycle iterations, checks convergence (rho = max|f - Au| ≤ tolerance), collects per-operation timing
- `src/kernel_gmg_vcycle.csl` — 1103-line state machine kernel with 28 states (DOWN_INIT through EXIT), callback-driven transitions via `f_trigger_state_machine()`
- `src/layout_gmg_vcycle.csl` — PE grid layout on 762×1172 WSE3 fabric, color/entrypoint allocation
- `src/blas.csl` — Basic linear algebra operations
- `python_gmg/gmgoscar.py` — Reference Python GMG solver (`SimpleGMG` class) for correctness verification
- `util.py` — Column-major data layout conversions between host (height,width,levels) and 1D device format

### State machine (kernel_gmg_vcycle.csl)
The V-cycle executes entirely on-device with minimal host intervention. Key state flow:
```
DOWN: INIT → SMOOTH_APPLY → SMOOTH_UPDATE → SMOOTH_CHECK (loop) →
      RESIDUAL_APPLY → RESIDUAL_COMPUTE → RESTRICT_REDUCE → RESTRICT_DIV → LEVEL_CHECK
COARSE: INIT → SMOOTH_APPLY → SMOOTH_UPDATE → SMOOTH_CHECK (loop)
UP:   INIT → EXPAND_Z → BCAST → INTERP_ADD →
      SMOOTH_APPLY → SMOOTH_UPDATE → SMOOTH_CHECK (loop) → LEVEL_CHECK
```
Each async operation (stencil, allreduce) triggers the next state via callback. Active PE selection: `is_active = (px % 2^level == 0) && (py % 2^level == 0)`.

### Artifact caching
`artifact_cache.json` maps output directory names to compiled artifact paths. `compile_and_run_wse3.py` checks the cache before compiling; if the artifact file exists, compilation is skipped. Jobs are launched on background threads (fire-and-forget).

### Per-operation timing
13+ timing arrays capture 48-bit hardware timestamps at each state transition, broken down by level:
- `timing_smooth`, `timing_residual`, `timing_restrict`, `timing_interp` (high-level)
- `timing_spmv_total/communication/compute` (stencil micro-benchmark)
- `timing_interp_expand_z/bcast_configure/bcast_to_all/interp_add` (interpolation breakdown)

### Output directory naming
`out_dir_S{size}x_L{levels}_M{max_ite}_P{pre}_P{post}_B{bottom}` — e.g., `out_dir_S128x_L7_M100_P6_P6_B100` = 128³ grid, 7 levels, max 100 iterations, 6 pre/post smooths, 100 bottom iterations. Shallow V-cycle variants use `shallow_out_dir_*`.

### Modified CSL libraries
`src/modified_csl_lib/` and `src/modified_csl_lib_hops/` contain forked versions of `stencil_3d_7pts` and `allreduce` with hop-based stencil support for coarse-grid operations at increasing PE distances.

### Documentation
Extensive docs in `docs/`: `README_STATE_MACHINE.md`, `README_VCYCLE.md`, `VCYCLE_STATE_MACHINE_SUMMARY.md`, `TIMING_GUIDE.md`, `QUICK_TIMING_REFERENCE.md`, `COMPARISON.md`.

## File Editing Rules

- When editing files, always confirm the exact file path with the user before making changes. Never edit files in directories you were told not to modify (e.g., csl_gmg, other agents' workspaces).
- When asked to fix or modify an existing script/tool, work with that existing code. Do not create a new replacement script unless explicitly asked.

## Cerebras / CSL Development

- Before starting implementation, clarify the existing architecture constraints (memcpy/runtime model, SdkLayout vs manual layout, codegen approach). Do not assume a new approach is compatible without checking.
- For CSL compilation: watch for common pitfalls - undefined pointers, missing export symbols, u32/u16 truncation, pe_x/pe_y param issues, and sub-rectangle memcpy constraints. Test compilation incrementally rather than making many changes at once.

## Project Conventions

- Primary language is Python. Visualization uses matplotlib. CSL kernel files use .csl extension. Always validate numeric computations (division ordering, unit scaling, fabric traffic counting) before plotting.

## Key Conventions

- Output directories (`out_dir_*`, `out/`) are gitignored compilation artifacts
- `--run-only` flag skips compilation when ELFs already exist
- Color IDs and entrypoint IDs must not collide across imported CSL modules
- Convergence: rho = max|f - Au| (infinity norm), checked after each V-cycle iteration on host
