# Roofline Analysis Methodology for GMG V-Cycle on WSE-3

This document describes every measurement, formula, and assumption used to build the roofline model for the Geometric Multigrid (GMG) V-cycle solver on Cerebras WSE-3. All analysis is for **1 V-cycle** (not N iterations).

---

## 1. Machine Parameters (CS-3 Golden Data)

All three roofline ceilings derive from a single frequency: **875 MHz**.

### 1.1 Compute Peak

```
PE_Peak = 1 FMAC/cycle × 2 FLOPs/FMAC × 875 MHz = 1.75 GFLOP/s per PE
```

FMAC (`a*b+c`) counts as 2 FLOPs (multiply + add), following the convention. All other f32 instructions (FADD, FMUL, FSUB, FNEG, FMAX) issue 1 per cycle = 1 FLOP/cycle. The peak assumes a pure FMAC stream.

### 1.2 Memory BW (SRAM)

```
PE_Mem_BW = 16 bytes/cycle read + 16 bytes/cycle write = 28.0 GB/s per PE
         = (4 f32/cycle read + 4 f32/cycle write) × 875 MHz
```

Each PE has 48 KB of SRAM with separate read and write ports. No cache hierarchy.

### 1.3 Fabric BW (inter-PE communication)

```
PE_Fabric_BW = 4 directions × 4 bytes/wavelet × 875 MHz = 14.0 GB/s per PE
```

Each PE's router has 4 bidirectional links to neighbor routers (N, S, E, W). Each link receives 1 wavelet (32 bits) per cycle. The total fabric receive bandwidth per PE is 4 wavelets/cycle.

**Note**: The router also has a 5th link (the "ramp") connecting to the CE at 1 wavelet/cycle. The ramp is tighter (3.5 GB/s) than the fabric (14.0 GB/s), but for the roofline we use the fabric bandwidth because we count inter-PE data movement, not the ramp bottleneck.

### 1.4 Ridge Points

```
Memory ridge  = PE_Peak / PE_Mem_BW    = 1.75 / 28.0  = 0.0625 FLOP/Byte
Fabric ridge  = PE_Peak / PE_Fabric_BW = 1.75 / 14.0  = 0.1250 FLOP/Byte
```

If AI > ridge → compute-bound on that resource. If AI < ridge → bandwidth-bound.

### 1.5 System Peaks Table


| Resource  | Formula               | Per PE       | 65,536 PEs (256²) | 893,064 PEs (full wafer) |
| --------- | --------------------- | ------------ | ----------------- | ------------------------ |
| Compute   | `N × 2 × 875 MHz`     | 1.75 GFLOP/s | 114.7 TFLOP/s     | 1.56 PFLOP/s             |
| Memory BW | `N × 28.0 GB/s`       | 28.0 GB/s    | 1.83 PB/s         | 25.0 PB/s                |
| Fabric BW | `N × 4 × 4 × 875 MHz` | 14.0 GB/s    | 917.5 TB/s        | 12.5 PB/s                |
| Mem ridge | `PE_Peak / PE_Mem_BW` | 0.0625       | 0.0625            | 0.0625                   |
| Fab ridge | `PE_Peak / PE_Fab_BW` | 0.1250       | 0.1250            | 0.1250                   |


Ridge points are the same at all scales because both numerator and denominator scale by the same N.

---

## 2. What We Measure (Counters)

All counters are **per PE(0,0)** for 1 V-cycle only (gated by `roofline_measure`). No division by iterations needed.

### 2.1 FLOP Counters

Runtime counters in the kernel track every floating-point instruction per level:

```csl
var count_fsub = @zeros([LEVELS]u32);    // 1 FLOP per element
var count_fmac = @zeros([LEVELS]u32);    // 2 FLOPs per element (mul + add)
var count_fmul = @zeros([LEVELS]u32);    // 1 FLOP
var count_fadd = @zeros([LEVELS]u32);    // 1 FLOP
var count_fneg = @zeros([LEVELS]u32);    // 1 FLOP
var count_fmov_mem = @zeros([LEVELS]u32);  // 0 FLOPs, @fmovs mem→mem (1 load + 1 store = 8 B)
var count_fmov_zero = @zeros([LEVELS]u32]; // 0 FLOPs, @fmovs zero-fill (0 loads + 1 store = 4 B)
var count_fmax = @zeros([LEVELS]u32);    // 1 FLOP
```

**Sources**: Instrumented in `kernel_gmg_vcycle.csl` (smooth, residual, restriction, interpolation, setup) and in `stencil_3d_7pts/wse3/pe.csl` (stencil compute — flushed to kernel via pointers). Also instrumented in `allreduce/pe.csl` (reduction FADD/FMAX — set via `setCounterPtrs`).

**Collection**: `memcpy_d2h` from PE(0,0), shape `[1, 1, LEVELS]`, dtype `u32`.

**1st V-cycle gating**: A `roofline_measure` flag in the kernel is `true` during V-cycle 1 and set to `false` after the first convergence check (when `k` transitions 0→1). All counter increments and timer start/stop calls check this flag. Subsequent V-cycles run without measurement overhead. **No division by iterations is needed** — counters directly contain 1 V-cycle values.

```python
FLOPs_L = sum(count[op][L] * FLOP_WEIGHT[op] for op in all_ops)  # already 1 V-cycle
```

### 2.2 Memory Traffic (Analytical from FLOP Counters)

Memory traffic is estimated from the instruction counts using a per-instruction model:


| Instruction | CSL builtin   | Mem Loads (f32) | Mem Stores (f32) | Total Bytes | Fabric Traffic |
| ----------- | ------------- | --------------- | ---------------- | ----------- | -------------- |
| FMAC        | `@fmacs`      | 3               | 1                | 16          | 0              |
| FMUL        | `@fmuls`      | 2               | 1                | 12          | 0              |
| FADD        | `@fadds`      | 2               | 1                | 12          | 0              |
| FSUB        | `@fsubs`      | 2               | 1                | 12          | 0              |
| FNEG        | `@fnegs`      | 1               | 1                | 8           | 0              |
| FMOV_MEM    | `@fmovs(m,m)` | 1               | 1                | 8           | 0              |
| FMOV_ZERO   | `@fmovs(m,0)` | 0               | 1                | 4           | 0              |
| FMAX        | `@fmaxs`      | 1               | 1                | 8           | 0              |
| FMOV32      | `@mov32` recv | 0               | 1                | 4           | 1 load = 4 B   |


```python
Mem_L = sum(count[op][L] * (loads[op] + stores[op]) * 4 for op in all_ops) + count_fmov32[L] * 4
```

**Three move instructions**:
- `FMOV_MEM` = `@fmovs(dest_sram, src_sram)`: SRAM-to-SRAM copy. 1 load + 1 store = 8 B memory.
- `FMOV_ZERO` = `@fmovs(dest_sram, 0.0)`: Zero-fill from immediate. 0 loads + 1 store = 4 B memory.
- `FMOV32` = `@mov32` receive from fabric. 1 fabric load = 4 B fabric. Plus 1 SRAM store = 4 B memory.
- `@mov32` sends (SRAM→fabric) are NOT counted — receive-side only convention.

**Known conservatism**: FMAC scalar operand is often in a register, not loaded from SRAM per element. True traffic is 2 loads + 1 store, not 3 + 1.

### 2.3 Fabric Traffic (Runtime Counter)

A dedicated counter tracks **data received from other PEs** via `@mov32`:

```csl
var count_fmov32 = @zeros([LEVELS]u32);  // @mov32 fabric receives per level
```

**Stencil** (`wse3/pe.csl`): `counter_fmov32` incremented before each async `@mov32` receive in `f_recv`. Only for active PEs (not inactive forwarding). Up to 4 directions (W, E, S, N), each `cur_length` elements.

**Allreduce** (`allreduce/pe.csl`): `local_count_fmov32_ptr` incremented via `setCounterPtrs` for every `@mov32` fabric receive in reduction and broadcast. Sends are NOT counted.

**What is counted**: Only data that originates from a different PE and is received by this PE. Receive-side only, no double counting. Inactive PE forwarding (router→router, no CE involvement) is NOT counted.

```python
Fabric_bytes_L = count_fmov32[L] * 4  # already 1 V-cycle, 1 wavelet = 4 bytes
```

### 2.4 Arithmetic Intensity

```python
Memory_AI_L  = FLOPs_L / Mem_L          # FLOP/Byte (SRAM)
Fabric_AI_L  = FLOPs_L / Fabric_bytes_L  # FLOP/Byte (inter-PE)
```

These are independent — memory and fabric are separate resources with separate ceilings.

---

## 3. What We Measure (Timers)

### 3.1 Timer Infrastructure

The kernel uses 12 hardware timer IDs (48-bit cycle counters at 875 MHz):


| Timer ID | Name                 | Scope                                                     |
| -------- | -------------------- | --------------------------------------------------------- |
| 0        | `timing_smooth`      | All pre-smooth + post-smooth iterations (SpMV + Jacobi)   |
| 1        | `timing_residual`    | Residual computation (SpMV + FSUB)                        |
| 2        | `timing_restrict`    | Restriction (allreduce + z-pair division)                 |
| 3        | `timing_interp`      | Interpolation (expand_z + broadcast + add)                |
| 4        | `timing_total`       | Total V-cycle wall clock (tic/toc around all iterations)  |
| 5-7      | `timing_interp_*`    | Interpolation micro-benchmarks                            |
| 8-9      | `timing_smooth_*`    | Smoothing micro-benchmarks (apply vs update)              |
| 10       | `timing_setup`       | `f_setup_level()`: DSD init, save/restore f, zero buffers |
| 11       | `timing_convergence` | Convergence check: SpMV + FSUB + allreduce MAX at L0      |


All per-level timers (IDs 0-3, 5-11) are arrays of `[LEVELS × tsc_size_words]`. Like counters, they are **gated by `roofline_measure`** — only active during V-cycle 1, so they directly contain 1 V-cycle values.

### 3.2 Per-Level Time

```
Time_L     = (smooth + residual + restrict + interp + setup + convergence)[L]  # directly 1 V-cycle
Conv_L     = convergence[L]              (non-zero only at L0)
Vcycle_L   = Time_L - Conv_L            (V-cycle work only, no diagnostic)
```

**Convergence check**: After V-cycle 1, a full SpMV + residual + allreduce MAX runs at L0 to check if `|r|_inf < tolerance`. This is a **solver diagnostic**, not part of the V-cycle algorithm.

**Convention**: The summary table reports **both** metrics:

- `GFLOP/s` and `%pk` = `FLOPs_L / Time_L` — wall-clock including convergence
- `GFLOP/s`* and `%pk*` = `FLOPs_L / Vcycle_L` — V-cycle only, excluding convergence

The roofline **plots** use V-cycle-only time (`Vcycle_L`) — this shows the kernel efficiency without the diagnostic overhead. For L1-L7, both columns are identical (convergence is zero). For L0:

- With convergence: ~24% of PE peak
- V-cycle only: ~38% of PE peak
- Difference: convergence consumes ~38% of L0's total time 
- WHY? It has a allreduce(col + row reduce and bcast to all PEs) + 1 spmv <----- It's costly, we must find ways to simplify it somehow.

### 3.3 Total V-Cycle Time

```
Total_1V_time = sum(Time_L for all levels)   (directly measured, NOT timing_total / iterations)
```

`timing_total` (ID 4) is NOT gated by `roofline_measure` — it measures the full solver wall clock across all iterations. 

### 3.4 Timer Collection

All timers collected from PE(0,0) via `memcpy_d2h`, converted from 48-bit cycle counts to microseconds:

```python
time_us = cycles / 0.875  # 875 MHz → 1 cycle = 1/875 us ≈ 1.143 ns
```

---

## 4. Three Roofline Experiments

All experiments use the same counters and timers from PE(0,0). They differ only in how we **scale** and what **ceiling** we compare against.

### 4.1 Experiment 1: Per PE(0,0) — "How efficient is each PE?"

**Scale**: No scaling. Raw PE(0,0) counters.

```
x-axis:  AI_L = FLOPs_L / Mem_L                    (or FLOPs_L / Fabric_bytes_L)
y-axis:  Achieved_L = FLOPs_L / Time_L             (FLOP/s)
ceiling: min(PE_Mem_BW × AI, PE_Peak)              (memory roofline)
         min(PE_Fabric_BW × AI, PE_Peak)            (fabric roofline)
% peak:  Achieved_L / PE_Peak
```

**What it shows**: Kernel efficiency at each multigrid level on a single PE. Coarse levels drop due to communication latency (not visible in the bandwidth roofline).

**Time used**: `Time_L` = per-level time (sum of 6 operation timers for that level).

### 4.2 Experiment 2: Active-PE System — "What throughput does each level achieve?"

**Scale**: Multiply PE(0,0) values by `Active_PEs_L = (grid_size / 2^L)²`.

```
x-axis:  AI_L                                       (same — scaling cancels)
y-axis:  (FLOPs_L × Active_PEs_L) / Time_L         (FLOP/s system)
ceiling: min(Active_PEs_L × PE_Mem_BW × AI, Active_PEs_L × PE_Peak)
         (different ceiling per level — each level has its own roofline line)
% peak:  y-axis / (Active_PEs_L × PE_Peak) = same as Experiment 1
```

**What it shows**: Absolute system throughput at each level alongside that level's own capacity. The % of peak is identical to Experiment 1 because active PEs cancel. The visual value is seeing the ceiling drop at coarser levels.

**Time used**: `Time_L` = same per-level time (all active PEs execute in lockstep).

### 4.3 Experiment 3: Full-Grid System — "How much of the allocated machine is used?"

**Scale**: Same system achieved as Experiment 2, but ceiling is **fixed** at `Fine_PEs = grid_size²`.

```
x-axis:  AI_L                                       (same)
y-axis:  (FLOPs_L × Active_PEs_L) / Time_L         (same achieved as Experiment 2)
ceiling: min(Fine_PEs × PE_Mem_BW × AI, Fine_PEs × PE_Peak)
         (fixed ceiling — same for all levels)
% peak:  y-axis / (Fine_PEs × PE_Peak)              (penalizes coarse levels for idle PEs)
```

**What it shows**: The multigrid idle-PE penalty. At L0, 100% of allocated PEs are active. At L7, only 4 out of 65,536 PEs are active (99.99% idle), so system utilization is near zero relative to the allocated peak.

**Time used**: `Time_L` = same per-level time. For the "Total" row: `Total_1V_time`.

### 4.4 Summary of What Differs


|         | Experiment 1 (Per PE) | Experiment 2 (Active PEs) | Experiment 3 (Full Grid) |
| ------- | --------------------- | ------------------------- | ------------------------ |
| FLOPs   | `FLOPs_L`             | `FLOPs_L × Active_PEs_L`  | `FLOPs_L × Active_PEs_L` |
| Traffic | `Mem_L` or `Fab_L`    | `× Active_PEs_L`          | `× Active_PEs_L`         |
| AI      | `FLOPs_L / Traffic_L` | same (cancels)            | same (cancels)           |
| Time    | `Time_L`              | `Time_L`                  | `Time_L`                 |
| Ceiling | 1 PE                  | `Active_PEs_L` PEs        | `Fine_PEs` PEs (fixed)   |
| % peak  | vs 1 PE               | vs active PEs (= Exp 1)   | vs allocated PEs         |


---

## 5. FLOP and Traffic Counting Details

### 5.1 FLOP Weights


| Instruction | CSL builtin       | Counter           | FLOPs | Memory Traffic           | Fabric Traffic |
| ----------- | ----------------- | ----------------- | ----- | ------------------------ | -------------- |
| FMAC        | `@fmacs`          | `count_fmac`      | 2     | 3 loads + 1 store = 16 B | 0              |
| FMUL        | `@fmuls`          | `count_fmul`      | 1     | 2 loads + 1 store = 12 B | 0              |
| FADD        | `@fadds`          | `count_fadd`      | 1     | 2 loads + 1 store = 12 B | 0              |
| FSUB        | `@fsubs`          | `count_fsub`      | 1     | 2 loads + 1 store = 12 B | 0              |
| FNEG        | `@fnegs`          | `count_fneg`      | 1     | 1 load + 1 store = 8 B   | 0              |
| FMOV_MEM    | `@fmovs(mem,mem)` | `count_fmov_mem`  | 0     | 1 load + 1 store = 8 B   | 0              |
| FMOV_ZERO   | `@fmovs(mem,0)`   | `count_fmov_zero` | 0     | 0 loads + 1 store = 4 B  | 0              |
| FMAX        | `@fmaxs`          | `count_fmax`      | 1     | 1 load + 1 store = 8 B   | 0              |
| FMOV32      | `@mov32` recv     | `count_fmov32`    | 0     | 0 loads + 1 store = 4 B  | 1 load = 4 B   |


**Three move instructions**:

- `FMOV_MEM` = `@fmovs(dest_sram, src_sram)`: SRAM-to-SRAM copy. 1 SRAM read + 1 SRAM write = 8 B memory.
- `FMOV_ZERO` = `@fmovs(dest_sram, 0.0)`: Zero-fill from immediate. 0 reads + 1 SRAM write = 4 B memory.
- `FMOV32` = `@mov32(dest_sram, src_fabric)`: Fabric receive. 1 fabric load = 4 B fabric. Plus 1 SRAM write = 4 B memory (data lands in SRAM).
- `@mov32` sends (SRAM→fabric) are NOT counted — receive-side only convention.

### 5.2 Where Counters Are Incremented

**In `kernel_gmg_vcycle.csl`**:


| Function                   | Level   | Counters incremented                                                       |
| -------------------------- | ------- | -------------------------------------------------------------------------- |
| `f_apply_operator`         | current | FMOV_ZERO (zero Au), then delegates to stencil                             |
| `f_jacobi_smooth`          | current | FSUB (r=f-Au), FMAC (u+=ω*r), FMOV_MEM (save u), FMOV_ZERO (zero inactive) |
| `f_residual`               | current | FSUB (r=f-Au), FMOV_ZERO (zero inactive)                                   |
| `f_restriction_division`   | current | FMOV_MEM ×2 (copy even→f, copy f_coarse→f), FADD, FMUL, FMOV_ZERO (inactive) |
| `f_interpolation_expand_z` | current | FMOV_MEM ×3 (save coarse, copy to even, copy to odd)                          |
| `f_interpolation_add`      | current | FADD (u += error), FMOV_ZERO (inactive)                                       |
| `f_setup_level`            | current | FMOV_ZERO (zero u, zero r), FMOV_MEM (save/restore f)                         |


**In `stencil_3d_7pts/wse3/pe.csl`** (flushed to kernel at callback):


| Function             | Counters                                         |
| -------------------- | ------------------------------------------------ |
| `spmv` init          | FMOV_ZERO (zero y_z buffer)                      |
| `laplacian_xy`       | FMAC (4 × cur_length), FNEG (boundary PEs only)  |
| `laplacian_z`        | FMAC (3 × zDim)                                  |
| `f_recv` (active PE) | FMOV32 (cur_length per direction received)        |


**In `allreduce/pe.csl`** (via `setCounterPtrs`):


| Function                         | Counters                                       |
| -------------------------------- | ---------------------------------------------- |
| `broadcast_to_all`               | FMOV32 (non-sender receives)                   |
| `broadcast_to_all_inside_window` | FMOV32 (window interior receives)              |
| `f_send_data` (reduce loop)      | FADD or FMAX + FMOV32 (accumulator receives)   |


### 5.3 PE(0,0) Corner PE Effects

PE(0,0) is at position (0,0) in the grid — a corner PE with boundary conditions on 2 sides:

- **Stencil**: Receives from 2 directions (east + south), not 4. Applies FNEG for west + north boundaries. So fmov32 is ~50% of an interior PE, but FNEG is ~2x higher.
- **Restriction**: PE(0,0) is the top-left accumulator — receives from neighbors during row/col reduce.
- **Interpolation**: PE(0,0) is the broadcast sender — does NOT receive (0 fabric loads for broadcast).

---

## 6. Formulas Reference

### Per 1 V-cycle, per level L, from PE(0,0) counters:

All values are directly from the 1st V-cycle (gated by `roofline_measure`, no division needed):

```
FLOPs_L       = Σ(count[op][L] × weight[op])
Mem_L         = Σ(count[op][L] × (loads[op] + stores[op]) × 4)
Fab_L         = count_fmov32[L] × 4                          (bytes)
Mem_AI_L      = FLOPs_L / Mem_L
Fab_AI_L      = FLOPs_L / Fab_L
Time_L        = (smooth + residual + restrict + interp + setup + convergence)[L]
Conv_L        = convergence[L]                                (non-zero only at L0)
Vcycle_L      = Time_L - Conv_L                               (V-cycle work only)
Achieved_L    = FLOPs_L / Vcycle_L                            (FLOP/s, used in plots)
% PE peak     = Achieved_L / PE_Peak × 100
```

### Scaling to system (Experiments 2 & 3):

```
Active_PEs_L  = (grid_size / 2^L)²
Sys_FLOPs_L   = FLOPs_L × Active_PEs_L
Sys_Achieved_L = Sys_FLOPs_L / Vcycle_L
% Active peak = Sys_Achieved_L / (Active_PEs_L × PE_Peak) × 100  = % PE peak (cancels)
% Grid peak   = Sys_Achieved_L / (grid_size² × PE_Peak) × 100    (penalizes idle PEs)
```

### Total V-cycle (all levels):

```
Total_FLOPs   = Σ FLOPs_L
Total_Mem     = Σ Mem_L
Total_Time    = sum(Vcycle_L for all levels)  (directly measured, 1st V-cycle)
Total_Achieved = Total_FLOPs / Total_Time
Sys_Total     = Σ(FLOPs_L × Active_PEs_L) / Total_Time
```

---

## 7. Known Issues and Notes

### 7.1 Convergence Check at L0

The convergence check (SpMV + FSUB + allreduce MAX) runs after V-cycle 1 at L0. Its FLOPs, fabric loads, and time are all counted in L0's accumulators. The table reports both:

- `GFLOP/s` / `%pk`: includes convergence (wall-clock honest)
- `GFLOP/s*` / `%pk*`: V-cycle only (convergence subtracted)

The roofline plots use **V-cycle only** (`Vcycle_L`) to show kernel efficiency without the diagnostic overhead.

### 7.2 1st V-cycle Measurement via `roofline_measure`

The `roofline_measure` flag gates all counter increments and timer start/stop calls. It is `true` during V-cycle 1 and set to `false` when `k` transitions 0→1 in `STATE_CONV_CHECK`. This means:

- Counters and timers directly contain 1 V-cycle values (no averaging)
- Subsequent V-cycles run without measurement overhead
- The solver still runs to convergence (iteration count is tracked via `counter_rho_check` which is not gated)
- `timing_total` (tic/toc around all iterations) is NOT gated — it gives the full solver wall time

**Simulator note**: The `roofline_measure` branching may cause stalls in the cycle-accurate simulator due to async dataflow interactions with runtime conditionals. It works correctly on hardware.

### 7.3 Setup Time (1st V-cycle specific)

Since only V-cycle 1 is measured, the setup at L0 is the `k=0` variant: save f only (no restore). 

### 7.4 Timer Coverage

The 6 per-level timers (gated to 1st V-cycle) capture the complete V-cycle. Their sum IS the 1V time — no averaging needed. The ~1.4% gap vs `timing_total / iterations` is state machine dispatch overhead.

---

## 8. File Map


| File                                                    | Role                                                                                                                                                              |
| ------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `src/kernel_gmg_vcycle.csl`                             | FLOP counters (`count_fmov_mem`, `count_fmov_zero`), fabric counter (`count_fmov32`), 12 timer IDs, `roofline_measure` gate, state machine                        |
| `src/layout_gmg_vcycle.csl`                             | Layout declarations for all exported symbols                                                                                                                      |
| `src/modified_csl_lib_hops/stencil_3d_7pts/wse3/pe.csl` | Stencil counters: FMAC, FNEG, FMOV (zero-fill), FMOV32 (fabric recv). Flushed to kernel via pointers.                                                             |
| `src/modified_csl_lib_hops/allreduce/pe.csl`            | Allreduce counters: FADD, FMAX, FMOV32 (fabric recv). Via `setCounterPtrs`. No FMOV (allreduce uses `@mov32`, not `@fmovs`).                                      |
| `run_gmg_vcycle.py`                                     | Collects raw counters/timers from PE(0,0), prints raw Table V (counts only). **No roofline computation** — that's in `roofline_analysis.py`.                      |
| `plots/roofline_analysis.py`                            | **Single source of truth for roofline**: parses response.txt, computes FLOPs/memory/fabric traffic/AI/achieved, generates 4-panel roofline plot + fabric heatmap. |
| `docs/ROOFLINE_METHODOLOGY.md`                          | This document                                                                                                                                                     |
|                                                         |                                                                                                                                                                   |


### Architecture

```
Device (CSL kernel)
  └─ Measures 1st V-cycle only (roofline_measure flag)
  └─ Raw counters: FSUB, FMAC, FMUL, FADD, FNEG, FMOV_MEM, FMOV_ZERO, FMAX, FMOV32
  └─ Raw timers: smooth, residual, restrict, interp, setup, convergence (per level)

run_gmg_vcycle.py
  └─ Collects raw counters + timers from PE(0,0)
  └─ Prints raw counts per level (no computation)
  └─ Writes response.txt

plots/roofline_analysis.py (single source of truth)
  └─ Parses response.txt
  └─ Computes: FLOPs, memory traffic, fabric traffic, AI, achieved FLOP/s
  └─ Generates: 4-panel roofline plot, fabric heatmap, summary tables
```

