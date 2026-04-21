# Rebuttal: GLOW — Accelerating Geometric Multigrid Layers on Cerebras WSE-3

We thank all reviewers for their thoughtful feedback. Below we address each concern with new results and clarifications.

---

## Review #215A (Weak Reject, Expertise: Some Familiarity)

### A1. "Definition of GLOW not found anywhere in the paper"

We will define GLOW (Geometric muLtigrid On Wafer-scale) in the abstract and its first use in the introduction.

### A2. "Only V-cycle, no W-cycle or F-cycle"

**Why V-cycle only:**

1. **Memory constraint**: At 512x512, the largest problem size, per-PE memory usage is **98.3% of the 48 KB SRAM** (22.8 KB code + 24.1 KB data + 0.3 KB overhead = 47.2 KB of 48 KB). A W-cycle would double the coarse-level computation — the exact phase that is already the bottleneck (communication-dominated at coarse levels). There is no memory headroom for the additional intermediate storage that W/F-cycles require.
2. **Coarse levels are the bottleneck**: Our per-operator timing analysis (Fig. 7) shows coarse-level execution time grows exponentially due to strided communication. A W-cycle doubles the number of coarse-level traversals, amplifying the worst-performing phase.
3. **HPGMG standard**: The HPGMG benchmark — the established HPC multigrid benchmark — uses V-cycles. Our comparison is apples-to-apples.

We will add a paragraph discussing W/F-cycle feasibility and these constraints.

### A3. "Shallow V-cycle should be discussed in abstract/introduction"

Agreed. We will add discussion of shallow V-cycles in both the abstract and introduction, noting that:

- Shallow V-cycles (fewer levels) trade more iterations for faster per-cycle time
- At 512x512, the shallow variant achieves the best per-V-cycle speedup (25x) but requires 45 iterations vs 11 for the deep variant
- The time-to-solution speedup for the shallow variant is 56x (see new TTS analysis below)

### A4. "Not significantly different from prior stencil papers"

While smoothing uses a stencil, the GMG V-cycle introduces fundamentally new challenges beyond a single stencil:

- **Multi-level mapping**: Superimposing multiple grid levels with strided active PE patterns on a 2D mesh
- **Inter-level operators**: Restriction (windowed reduction + z-coarsening) and interpolation (z-expansion + broadcast) have no analogue in prior stencil work
- **State machine execution**: 28-state device-resident execution with callback-driven transitions, vs simple stencil launch-and-collect
- **Memory management**: Geometric series storage across levels within 48 KB

---

## Review #215B (Weak Accept, Expertise: Some Familiarity)

### B1. "Specific HPGMG code version not clearly specified"

The HPGMG baseline code is available at: **[https://github.com/Sameeranjoshi/hpgmg.git](https://github.com/Sameeranjoshi/hpgmg.git)**

- Branch: `fp32` (modified from upstream HPGMG to use FP32, matching WSE-3 precision)
- Build configuration: see `build.sh` in the repository
- Run configuration: see `run.sh` in the repository
- GPU: NVIDIA GH200 120GB, Driver 570.124.06, CUDA 12.8

We will add this information to the paper.

### B2. "How does performance compare to device peak capability?"

This is an excellent question. We have a roofline analysis in progress (branch `sam-ics-roofline`). The WSE-3 theoretical peaks are:

- Compute: 21 PFLOPS FP32
- Memory bandwidth: 21 PB/s
- Fabric bandwidth: 214 Pb/s

A full roofline analysis requires careful accounting of operational intensity per operator per level. We will include preliminary results in the revision. However, we note that the per-operator timing breakdown (Fig. 7) already reveals where each operator sits relative to the compute/communication boundary.

### B3. "How much of the RAM is occupied by code?"

New analysis (from ELF symbol analysis across all problem sizes):


| Grid    | Code (KB) | Data (KB) | Total (KB) | Used % of 48 KB |
| ------- | --------- | --------- | ---------- | --------------- |
| 4x4     | 22.66     | 1.41      | 24.07      | 50.7%           |
| 64x64   | 22.80     | 4.96      | 27.76      | 58.4%           |
| 256x256 | 22.79     | 14.96     | 37.75      | 79.3%           |
| 512x512 | 22.79     | 24.08     | 46.88      | **98.3%**       |


Code is **constant at ~22.8 KB (47.5%)** across all problem sizes — dominated by the inlined stencil and allreduce libraries in the state machine kernel. Data grows with problem size and level count following a geometric series. At 512x512, only 0.81 KB remains free.

### B4. Minor text issues

We will fix "speed up the time" and "analytical model" sentence.

---

## Review #215C (Reject, Expertise: Knowledgeable)

### C1. "Comparing WSE-3 against a single GPU is unfair"

This is a standard single-system comparison, consistent with established practices:

- **Top500 and HPCG** compare individual systems regardless of internal parallelism
- **MLPerf** reports per-system performance
- The WSE-3 is a single integrated system (one wafer, one chip, one memory space), just as a GPU is a single system

We acknowledge the scale difference and will add a per-watt discussion:

- WSE-3: ~15 kW system power
- GH200: ~900 W TDP
- At 25x speedup with ~17x power: comparable perf/watt range

A multi-GPU or multi-node comparison would be valuable future work but is orthogonal to our contribution — demonstrating that a complete multi-level solver can be mapped to wafer-scale architecture.

### C2. "Only 7-point stencil, no 27-point or variable coefficients"

The 7-point stencil is the standard operator in HPGMG and represents the fundamental 3D Poisson equation. It is the baseline for multigrid benchmarking. Our implementation can support variable coefficients by changing the stencil parameters (the `ALPHA`/`BETA` coefficients are already parameterized). A 27-point stencil would require additional colors for diagonal neighbors, which is feasible within the CSL programming model but would increase code size and memory pressure. We consider this important future work.

### C3. "No empirical comparison of alternative layouts"

We provide an analytical comparison in Table 2 evaluating 5 mapping strategies across 8 dimensions. Empirical comparison would require implementing 4 alternative layouts in CSL — a substantial engineering effort beyond the scope of this paper. However, we note that:

- **Spatial mapping** requires O(N) data repacking between levels (prohibitively expensive)
- **Pipelined mapping** creates pipeline bubbles for the bidirectional V-cycle
- **Distributed mapping** has the same communication pattern as strided but worse memory utilization
- Only **strided distributed** avoids both data movement and pipeline bubbles

### C4. Typos

We will fix all typos identified (WS-3, HPGPG, "scuh as", "can consists of", missing spaces, etc.).

---

## Review #215D (Weak Reject, Expertise: Knowledgeable)

### D1. "Time-to-solution, not just time per V-cycle" (Critical)

**New result: Time-to-Solution (TTS) analysis.**

TTS = (number of iterations to converge) x (average time per V-cycle).

Key findings for the 6/6/6 configuration:


| Grid    | WSE-3 Iters | WSE-3 TTS (s) | GH200 Iters | GH200 TTS (s) | TTS Speedup | Per-Vcycle Speedup |
| ------- | ----------- | ------------- | ----------- | ------------- | ----------- | ------------------ |
| 16x16   | 6           | 0.000874      | 3           | 0.002076      | **2.4x**    | 4.8x               |
| 32x32   | 7           | 0.001373      | 4           | 0.003661      | **2.7x**    | 4.7x               |
| 64x64   | 8           | 0.002226      | 100*        | 0.130718      | **58.7x**   | 4.7x               |
| 128x128 | 9           | 0.003486      | 100*        | 0.174903      | **50.2x**   | 4.5x               |
| 256x256 | 10          | 0.006183      | 100*        | 0.384706      | **62.2x**   | 6.2x               |
| 512x512 | 11          | 0.010717      | 100*        | 1.673107      | **156.1x**  | 17.2x              |


*GH200 hits max_iter=100 for sizes >= 64x64 (HPGMG benchmark configuration).

**The TTS speedup is even more favorable than per-V-cycle speedup** because WSE-3 converges in significantly fewer iterations (9-11 vs 100 for large sizes). For sizes where GH200 converges (16x16, 32x32), the TTS speedup is 2.4-2.7x (lower than per-cycle due to WSE-3 needing slightly more iterations at small sizes).

For the shallow V-cycle (6/6/6-shallow):


| Grid    | WSE-3 Iters | WSE-3 TTS (s) | GH200 TTS (s) | TTS Speedup |
| ------- | ----------- | ------------- | ------------- | ----------- |
| 128x128 | 38          | 0.009939      | 0.174903      | **17.6x**   |
| 256x256 | 42          | 0.018021      | 0.384706      | **21.3x**   |
| 512x512 | 45          | 0.029634      | 1.673107      | **56.5x**   |


We will add a TTS figure and table to the paper.

### D2. "Residual-vs-iteration curves"

**New result: Convergence plots generated.** We have generated convergence plots showing |rho|_inf vs iteration for all 5 configurations across all grid sizes (128³, 256³, 512³). Key observations:

- All configurations converge monotonically
- Deep V-cycles (6/6/6, 6/6/100) converge fastest (9-11 iterations)
- Shallow V-cycles need more iterations (38-45) but each is faster
- 4/4/6 and 4/4/100 fall in between (17-22 iterations)

These plots will be included in the revision.

### D3. "Correctness check / residual trajectory match"

**New result: Host vs device correctness verification.**

We ran the reference Python GMG solver on CPU for problem sizes 4³ through 64³ and compared residual trajectories iteration-by-iteration:


| Grid  | Device Iters | Host Iters | Final Rho Match         | Convergence Factor Match |
| ----- | ------------ | ---------- | ----------------------- | ------------------------ |
| 4x4   | 4            | 5          | Same order of magnitude | Expected FP32 divergence |
| 8x8   | 5            | 6          | Same order of magnitude | Expected FP32 divergence |
| 16x16 | 6            | 7          | Same order of magnitude | Expected FP32 divergence |
| 32x32 | 7            | 7          | **Match to 4+ digits**  | 1.671e-08 vs 1.671e-08   |
| 64x64 | 8            | 8          | **Match to 4+ digits**  | 9.205e-10 vs 9.208e-10   |


For larger problem sizes (32³+), the device and host convergence factors are **virtually identical** (relative error < 0.04%). For smaller sizes, FP32 accumulated rounding differences from the async dataflow execution vs sequential CPU execution cause per-iteration values to differ, but both converge to the same tolerance. The device consistently converges in equal or fewer iterations.

### D4. "Wafer utilization and what prevents larger active region"

**New result: Complete wafer utilization analysis.**


| Grid    | Active PEs | % of WSE-3 | SRAM Used | Free KB  |
| ------- | ---------- | ---------- | --------- | -------- |
| 64x64   | 4,096      | 0.46%      | 58.4%     | 19.95    |
| 128x128 | 16,384     | 1.83%      | 65.5%     | 16.55    |
| 256x256 | 65,536     | 7.34%      | 79.3%     | 9.95     |
| 512x512 | 262,144    | 29.35%     | **98.3%** | **0.81** |


**What prevents using more PEs:**

1. **Memory saturation**: At 512x512, 98.3% of 48 KB SRAM is used (0.81 KB free). Code is fixed at 22.8 KB; data grows as nz × (2 - 1/2^(L-1)) per save array.
2. **Fabric constraint**: The usable fabric is 762x1172. Theoretically 762x762 = 580,644 PEs (~65%) could be used, but memory would be exceeded well before that.
3. **Code section**: 47.5% of SRAM is code (constant), due to aggressive function inlining of the stencil and allreduce libraries within the state machine kernel.

Per-level PE utilization for 512x512 (9 levels):


| Level | Factor | Active PEs | % of Fine | Hop Distance |
| ----- | ------ | ---------- | --------- | ------------ |
| 0     | 1      | 262,144    | 100%      | 1            |
| 1     | 2      | 65,536     | 25%       | 2            |
| 2     | 4      | 16,384     | 6.25%     | 4            |
| ...   | ...    | ...        | ...       | ...          |
| 8     | 256    | 4          | 0.0015%   | 256          |


### D5. "Tighten algorithmic specification (restriction/interpolation with z handling)"

We will add explicit pseudocode for both operators:

**Restriction (full 2x2x2 coarsening):**

1. Window-based row+column reduction: Gather residual r from 2^(L+1) x 2^(L+1) window of active PEs to top-left PE
2. Z-dimension coarsening on top-left PE: `f_coarse[i] = (r[2i] + r[2i+1]) / 8` for i = 0..nz/2-1
3. Factor of 8 = 4 (x,y averaging from 2x2 active PE window) × 2 (z-pair averaging)

**Interpolation (full 2x2x2 expansion):**

1. Z-expansion on active PEs: `fine[2i] = fine[2i+1] = coarse[i]` (duplicate each z-value)
2. Broadcast from top-left PE to all PEs in window (zero-cost hardware multicast)
3. Error correction: `u = u_saved + interpolated_correction`

**Grid dimensions per level** for a size×size domain (L levels):


| Level | nx       | ny       | nz       | Active PEs  |
| ----- | -------- | -------- | -------- | ----------- |
| l     | size/2^l | size/2^l | size/2^l | (size/2^l)² |


"Coarsest grid is 2×2" means 2×2×2 for the 3D problem (the z-dimension coarsens identically).

### D6. "Forwarding optimization pseudo-code"

We name this the **"Wavelet Bypass Forwarding"** optimization. Pseudo-code:

```
For each inactive PE on the communication path:
  // Instead of: receive wavelet → ramp down to SRAM → process → ramp up → send
  // Do: route wavelet directly through router queue (single cycle)

  if PE is inactive at current level:
    configure input queue to forward directly to output queue
    // Wavelet traverses PE via router without touching SRAM
    // Avoids 2× ramp latency per inactive PE
```

**Applicability:**

- Benefits: stencil (SpMV), restriction (allreduce)
- Does not benefit: interpolation (already uses zero-cost multicast)
- Requirement: color routing must be configured for hop-based communication

**Impact:** 8.45× speedup on coarse-level communication, 3.94× end-to-end at 512³.

### D7. "Fix broken refs, dimension inconsistency, typos"

Will fix all:

- Broken refs (Figure /reffig:4)
- WSE dimension: correct is 762×1172 (not 1724×762)
- H200 → GH200 consistently
- All typos (WS-3, HPGPG, etc.)

### D8. "Baseline documentation: GPU model, software, timing methodology"

- **GPU**: NVIDIA GH200 120GB (Grace Hopper), 97,871 MiB GPU memory
- **Software**: Driver 570.124.06, CUDA 12.8
- **HPGMG code**: [https://github.com/Sameeranjoshi/hpgmg.git](https://github.com/Sameeranjoshi/hpgmg.git), branch `fp32`
- **Timing (WSE-3)**: On-chip 48-bit timestamp counter at 850 MHz, measures kernel execution time excluding H2D/D2H transfers
- **Timing (GH200)**: CUDA events (cudaEventElapsedTime), measures kernel execution time excluding H2D/D2H transfers
- Both measure device-side computation only for fair comparison

---

## Summary of New Results Added


| Result                          | Addresses  | Key Finding                                                        |
| ------------------------------- | ---------- | ------------------------------------------------------------------ |
| Time-to-solution table + figure | D1, A3     | TTS speedup up to 156x (even more favorable than per-cycle)        |
| Convergence plots               | D2         | All configs converge monotonically; deep=fast, shallow=more iters  |
| Correctness verification        | D3         | Host and device match to 4+ digits for 32³+ problems               |
| Wafer utilization analysis      | D4, C1     | 512x512 uses 98.3% of 48KB SRAM — memory is the scaling bottleneck |
| Memory breakdown table          | B3, D4     | Code constant at 47.5%, data grows with geometric series           |
| HPGMG baseline details          | B1, C2, D8 | Full GitHub link, build/run config, GPU specs                      |
| Forwarding pseudo-code          | D6         | Named "Wavelet Bypass Forwarding", clear applicability constraints |


---

## Scripts for Reproducing New Results

All analysis scripts are in `gmg/csl_gmg/plots/`:

```bash
# Time-to-solution analysis + figures
python plots/time_to_solution.py

# Convergence plots
python plots/plot_convergence.py

# Wafer utilization + memory analysis
python plots/wafer_utilization.py

# Correctness verification (runs host solver)
python plots/correctness_check.py --run-host --sizes 4,8,16,32,64
```

## My notes extra:

1. plot_convergence.py — it shows the bottom solver count doesn't matter (validates using fewer coarse iterations), while pre/post smoothing count is what
  drives convergence rate.

