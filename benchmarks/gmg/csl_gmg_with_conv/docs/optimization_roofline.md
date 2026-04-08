# Optimization Opportunities for GMG V-Cycle on WSE-3

Based on the roofline analysis of the 256x256x256 V-cycle (8 levels, pre/post=6, bottom=6).
All data is from the 1st V-cycle measurement (roofline_measure gated).

See `ROOFLINE_METHODOLOGY.md` for measurement methodology and machine parameters.

---

## 1. Current Performance Profile

### Per-Level Timing (1st V-cycle, PE(0,0))

| Level | nz | Active PEs | Smooth (us) | Residual (us) | Restrict (us) | Interp (us) | Setup (us) | Conv (us) | Total (us) | % PE Peak* |
|---|---|---|---|---|---|---|---|---|---|---|
| L0 | 256 | 65,536 | ~87 | ~6 | ~2 | ~2 | ~5 | ~61 | ~165 | 37.7% |
| L1 | 128 | 16,384 | ~59 | ~4 | ~3 | ~2 | ~3 | 0 | ~72 | 25.6% |
| L2 | 64 | 4,096 | ~44 | ~3 | ~3 | ~2 | ~3 | 0 | ~57 | 16.7% |
| L3 | 32 | 1,024 | ~38 | ~3 | ~4 | ~1 | ~3 | 0 | ~52 | 9.8% |
| L4 | 16 | 256 | ~37 | ~3 | ~6 | ~1 | ~3 | 0 | ~53 | 5.3% |
| L5 | 8 | 64 | ~46 | ~3 | ~11 | ~1 | ~3 | 0 | ~69 | 2.5% |
| L6 | 4 | 16 | ~66 | ~5 | ~20 | ~1 | ~3 | 0 | ~103 | 1.1% |
| L7 | 2 | 4 | ~53 | 0 | 0 | 0 | ~2 | 0 | ~55 | 0.2% |

*V-cycle only (excluding convergence). Convergence adds ~61 us to L0.

### Key Bottlenecks

1. **Coarse-level communication** (L4-L7): Smoothing time increases despite fewer FLOPs because stencil communication traverses 16-128 hop distances
2. **Restriction at coarse levels**: Grows from 2 us (L0) to 20 us (L6) — allreduce over large PE distances
3. **Convergence check**: 61 us per V-cycle (37% of L0's total time) for a diagnostic SpMV + allreduce MAX

---

## 2. Optimization Opportunities

### OPT-1: Reduce Convergence Check Frequency (Easy, High Impact)

**Current**: Full SpMV + residual + allreduce MAX at L0 after every V-cycle iteration. Cost: ~61 us = 37% of L0 time.

**Proposal**: Check every N iterations (e.g., every 2-3), or use a cheaper convergence indicator.

**Impact**: For 10 iterations, current cost = 10 × 61 = 610 us. Checking every 3 = ~4 checks, saving ~366 us = ~6% of total solver time.

**Effort**: Low — modify `STATE_CONV_CHECK` to skip the SpMV on non-check iterations (check `k % N == 0`).

### OPT-2: Variable Smoothing Iterations per Level (Easy, Medium Impact)

**Current**: 6 pre-smooth + 6 post-smooth at every level.

**Observation**: At coarse levels (L5-L7), smoothing is communication-dominated (64-87% of smooth time is communication). Each smooth iteration requires a full SpMV, and additional smoothing has diminishing returns on tiny grids (nz=4,8).

**Proposal**: Fewer smoothing iterations at coarse levels:
- L0-L2: 6 pre + 6 post (current)
- L3-L4: 4 pre + 4 post
- L5-L6: 2 pre + 2 post
- L7 (bottom): 6 (current)

**Impact**: At L6 (nz=4), smooth = 66 us for 12 iterations. Reducing to 4 saves ~44 us. Across L3-L6: could save ~80-100 us = ~13-16% of V-cycle time. Convergence rate may degrade (more V-cycle iterations needed).

**Effort**: Low — parameterize per-level smooth counts.

### OPT-3: Allreduce Tree Reduction (Medium, Medium Impact)

**Current**: The allreduce for restriction uses a sequential chain — the accumulator PE receives from all participating PEs one by one. In `f_send_data()` (allreduce/pe.csl), the last PE loops `reduce_pes` times, each iteration receiving one wavelet from the chain.

For a 2×2 restriction window (stride doubles each level), `reduce_pes = 1` per direction. But the row/column reductions at the window boundaries involve forwarding across `2^(L+1)` hops. At L6 (stride=64), data traverses 64 inactive PEs before reaching the accumulator.

**What's already optimized**: The forwarding optimization from the ICS GLOW paper (Section 6.3) is already implemented in `modified_csl_lib_hops` — inactive PEs forward directly from router input queue to router output queue without touching the CE/SRAM. This reduces per-hop latency from ~5 cycles to ~2 cycles.

**Remaining opportunity**: The broadcast phase (`broadcast_to_all_inside_window`) uses a single-source broadcast from the top-left PE. For large windows at coarse levels, this is also latency-bound by hop count. A tree-based broadcast could reduce latency from O(width) to O(log width).

**Impact**: Restriction at L6 = 20 us. A tree reduction could reduce to ~5-8 us. Across all levels: ~15-20 us savings = ~3% of V-cycle time.

**Effort**: Medium — requires restructuring the allreduce state machine. The CSL fabric constraints (limited colors, wavelet routing) may complicate tree topologies.

### OPT-4: Overlap laplacian_z with Communication (Medium, Low Impact at fine levels)

**Current**: The stencil SpMV executes sequentially:
```
SEND halo → RECV halo → laplacian_xy → laplacian_z
```
`laplacian_z` uses only local z-neighbor data (no fabric communication) but waits for all communication to finish.

**Proposal**: Execute `laplacian_z` concurrently with SEND/RECV. It requires only local SRAM data.

**Impact**: At L0, `laplacian_z` cost ≈ 768 FMACs ÷ 875 MHz ≈ 0.88 us. Communication = ~27 us. The overlap benefit is small at fine levels where compute dominates. At L5-L7 where communication is 64-87% of time, overlapping could recover some compute behind the wait.

**Effort**: Medium — requires restructuring the stencil state machine to launch `laplacian_z` before SEND/RECV completes.

### OPT-5: Reduce FMOV Overhead (Medium, Low Impact)

**Current FMOV breakdown per V-cycle at L0**:

| Source | Type | Count | Bytes |
|---|---|---|---|
| Zero-fill Au (per SpMV) | FMOV_ZERO | 14 × 256 = 3,584 | 14,336 |
| Zero-fill y_z (stencil, per SpMV) | FMOV_ZERO | 14 × 256 = 3,584 | 14,336 |
| Save u_smooth (per smooth iter) | FMOV_MEM | 12 × 256 = 3,072 | 24,576 |
| Zero-fill r/u (setup) | FMOV_ZERO | ~768 | 3,072 |
| Save/restore f (setup) | FMOV_MEM | ~512 | 4,096 |
| Restriction copies | FMOV_MEM | ~256 | 2,048 |
| Interpolation copies | FMOV_MEM | ~384 | 3,072 |

**Potentially eliminable**:
- **save_u_smooth** (3,072 elements per V-cycle at L0): After every smoothing iteration, u is copied to `save_u_smooth` for later use in interpolation correction. Could be eliminated by swapping buffer pointers instead of copying, or computing interpolation correction in-place.
- **Restriction double-copy**: Copies even z-indices to intermediate buffer, then copies back. Could use DSD stride tricks to accumulate in-place.

**Impact**: Eliminating save_u_smooth saves ~24 KB of SRAM bandwidth per V-cycle at L0. This is ~4% of total memory traffic. AI improves slightly.

**Effort**: Medium — requires rethinking the interpolation-add data flow.

---

## 3. Memory Footprint and 512×512×512 Constraint

### Current Memory Layout (per PE, 256×256×256 problem)

```
Working arrays (fixed size = MAX_ZDIM = 256):
  u[256]                    = 1,024 B
  f[256]                    = 1,024 B
  r[256]                    = 1,024 B
  Au[256]                   = 1,024 B
  Subtotal:                   4,096 B

Superimposed save arrays (all levels packed):
  TOTAL_GMG_SIZE = 256 + 128 + 64 + 32 + 16 + 8 + 4 + 2 = 510 elements
  save_u_smooth[510]        = 2,040 B
  save_f_down[510]          = 2,040 B
  Subtotal:                   4,080 B

Stencil buffers (BLOCK_SIZE = 256):
  west_buf[256]             = 1,024 B
  east_buf[256]             = 1,024 B
  south_buf[256]            = 1,024 B
  north_buf[256]            = 1,024 B
  Subtotal:                   4,096 B

Timing + counters + misc:    ~2,000 B
Code section:                ~23,000 B (47% of 48 KB)

TOTAL:                       ~37 KB of 48 KB (~77% utilization)
```

### Why 512×512×512 Doesn't Fit

For 512³ with BLOCK_SIZE=256:
- `TOTAL_GMG_SIZE = 512 + 256 + ... + 2 = 1,022` → save arrays = 8,088 B (2× current)
- Working arrays at MAX_ZDIM=512: u/f/r/Au = 8,192 B (2× current)
- Stencil buffers stay at BLOCK_SIZE=256: 4,096 B (unchanged)
- Total: ~45+ KB → **exceeds 48 KB** when code section (~23 KB) is included

### Possible Memory Optimizations

1. **Eliminate save_u_smooth**: Store only the post-smooth u at the current level, not all levels. Use an in-place correction during interpolation. Saves `TOTAL_GMG_SIZE × 4` = 2,040 B for 256³, 4,088 B for 512³.

2. **Reduce timing arrays**: 12 timer IDs × 8 levels × 6 bytes = 576 B. For production runs (not profiling), these could be eliminated. Saves ~2 KB.

3. **Smaller BLOCK_SIZE for 512**: Using BLOCK_SIZE=128 instead of 256 halves the stencil halo buffers from 4,096 B to 2,048 B, at the cost of more communication rounds.

4. **Code size reduction**: The code section is ~23 KB (47%). Removing debug/timing code and simplifying the state machine could recover several KB. The `--inline-threshold=256 --unroll-threshold=256` compiler flags aggressively expand code — reducing these would shrink code at the cost of some performance.

5. **Compress save_f_down**: Currently stores f for all levels. If the algorithm could recompute f at each level (from the original RHS + restriction), save_f_down could be eliminated entirely. This would require algorithmic changes to the V-cycle restore logic.

### Memory Budget for 512³

| Component | 256³ (current) | 512³ (projected) | With OPT |
|---|---|---|---|
| u, f, r, Au | 4,096 B | 8,192 B | 8,192 B |
| save_u_smooth | 2,040 B | 4,088 B | 0 B (eliminated) |
| save_f_down | 2,040 B | 4,088 B | 4,088 B |
| Stencil buffers | 4,096 B | 4,096 B | 2,048 B (BLOCK=128) |
| Timing + counters | 2,000 B | 2,000 B | 500 B (minimal) |
| Code | 23,000 B | 23,000 B | 20,000 B (reduced) |
| **Total** | **37,272 B** | **45,464 B** | **34,828 B** |
| **Available** | **48,000 B** | **48,000 B** | **48,000 B** |
| **Fits?** | Yes | **No** | **Yes** |

Eliminating `save_u_smooth` + reducing BLOCK_SIZE + trimming timing arrays could make 512³ feasible.

---

## 4. Optimization Priority Matrix

| # | Optimization | Wall Time | Memory | Effort | Priority |
|---|---|---|---|---|---|
| OPT-1 | Convergence check every N | ~6% solver time | — | Low | **Do first** |
| OPT-2 | Variable smooth iters/level | ~13-16% V-cycle | — | Low | **Do first** |
| OPT-3 | Tree allreduce/broadcast | ~3% V-cycle | — | Medium | Do second |
| OPT-4 | Overlap laplacian_z with comm | ~1-5% V-cycle | — | Medium | Do second |
| OPT-5 | Reduce FMOV (save_u_smooth) | ~4% AI improvement | -2 KB | Medium | Do second |
| MEM-1 | Eliminate save_u_smooth | — | **-4 KB** (512³) | Medium | **For 512³** |
| MEM-2 | Reduce BLOCK_SIZE for 512 | Slight slowdown | **-2 KB** | Low | **For 512³** |
| MEM-3 | Trim timing arrays | — | **-1.5 KB** | Low | **For 512³** |
| MEM-4 | Reduce code size | — | **-3 KB** | Low | **For 512³** |

**Quick wins** (OPT-1 + OPT-2): ~20% wall time reduction with minimal code changes.

**For 512³** (MEM-1 + MEM-2 + MEM-3 + MEM-4): Combined savings of ~10.5 KB, bringing 512³ from 45.5 KB to ~35 KB (fits in 48 KB).
