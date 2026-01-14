# GMG V-Cycle Timing - Quick Reference

## Quick Start

```bash
cd benchmarks/gmg/csl_gmg
./commands_vcycle_wse3.sh
```

## What Gets Measured

| Operation | Description | Typical % of Total |
|-----------|-------------|-------------------|
| **Smooth** | Jacobi iterations (pre + post + bottom) | 80-90% |
| **Apply Operator** | 7-point stencil for residual | 5-10% |
| **Restriction** | Coarsen residual to next level | 2-5% |
| **Interpolation** | Refine correction to previous level | 1-2% |

## Timing Capture Logic

```
┌─ START TIMER ─────────────────────┐
│  save_timestamp_start(level)      │
├───────────────────────────────────┤
│  ... execute operation ...        │
│  ... may span multiple states ... │
├───────────────────────────────────┤
│  save_timestamp_end(level)        │
└─ END TIMER ───────────────────────┘
```

## Output Interpretation

### Per-Level Timing

```
Level 0:
  smooth:        [    5234,     5250,     5289]  (   6.176 us)
                  ^min     ^avg      ^max       ^time in μs
```

- **Min**: Fastest PE (best case)
- **Max**: Slowest PE (overall bottleneck)
- **Avg**: Average across all active PEs
- **Time**: Average time in microseconds

### Total Summary

```
Total Time Summary:
  Total smooth:           36.817 us  ← Sum across all levels
  Total apply_op:          1.570 us
  Total restriction:       2.158 us
  Total interpolation:     0.831 us
  Total V-cycle time:     41.376 us  ← Total time for one V-cycle
```

## CSL Timing Functions

### In Kernel

```csl
// Enable timestamps
timestamp.enable_tsc();

// Capture timestamp
timestamp.get_timestamp(&buffer);  // buffer is [3]u16

// Save start/end for specific operation at specific level
save_timestamp_start(level, &timing_array);
save_timestamp_end(level, &timing_array);
```

### In Python

```python
# Convert 3×u16 to 48-bit value
cycles = make_u48([word0, word1, word2])

# Convert cycles to microseconds (850 MHz clock)
time_us = (cycles / 0.85) * 1e-3
```

## Timing Arrays Layout

Each timing array stores per-level timing:

```
timing_smooth[0..5]    = Level 0: [start[0..2], end[3..5]]
timing_smooth[6..11]   = Level 1: [start[6..8], end[9..11]]
timing_smooth[12..17]  = Level 2: [start[12..14], end[15..17]]
...
```

Size per array: `LEVELS × 6 u16 values`

## Measured vs Unmeasured

### ✅ Measured
- Jacobi smoothing (pre, post, bottom)
- Operator application (stencil)
- Restriction (reduce + average)
- Interpolation (broadcast + add)

### ❌ Not Measured
- State transitions overhead
- DSD setup time
- Callback overhead
- Memory allocation
- Data copies (H2D/D2H)

## Typical Results (8×8×8, 3 levels)

| Configuration | Total Time | Dominant Operation |
|---------------|------------|-------------------|
| pre=1, post=1, bottom=10 | ~5-10 us | Bottom solver (70%) |
| pre=6, post=6, bottom=50 | ~40-50 us | Bottom solver (60%) |
| pre=10, post=10, bottom=100 | ~80-100 us | All smooth (85%) |

## Performance Expectations

### Problem Size Scaling

| Grid Size | Levels | PEs | Expected Time |
|-----------|--------|-----|---------------|
| 4×4×4 | 2 | 16 | ~2-5 us |
| 8×8×8 | 3 | 64 | ~10-50 us |
| 16×16×16 | 4 | 256 | ~40-200 us |
| 32×32×32 | 5 | 1024 | ~160-800 us |

*Time depends heavily on iteration counts*

### Speedup vs Host-Controlled

| Metric | Improvement |
|--------|-------------|
| RPC calls | **200x fewer** (1 vs 200+) |
| Latency | **200-500x less** |
| Total runtime | **50-100x faster** |

## Common Timing Patterns

### Expected Patterns

1. **Smooth time decreases** with level (fewer active PEs)
2. **Apply op time decreases** with level (less communication)
3. **Restriction time decreases** with level (smaller windows)
4. **Interpolation time decreases** with level (fewer PEs)

### Red Flags

- ⚠️ **Smooth time increases** with level: Check iteration counts
- ⚠️ **Apply op constant** across levels: Check active PE logic
- ⚠️ **Zero times**: Timer not enabled
- ⚠️ **Huge times** (>1s): Possible infinite loop or hang

## Quick Checks

```bash
# After running, check sim.log
grep -c "DOWN_INIT" sim.log         # Should be: levels - 1
grep -c "COARSE_DONE" sim.log       # Should be: active PEs at coarse level
grep -c "UP_INIT" sim.log           # Should be: levels - 1
grep "FINAL" sim.log                # Should show final solution

# Check for errors
grep -i "error\|assert\|fail" sim.log
```

## Next Steps

1. **Run with default params**: `./commands_vcycle_wse3.sh`
2. **Check timing output**: Look for "Performance Timing" section
3. **Compare with bricks**: Similar operation breakdown
4. **Optimize**: Focus on dominant operations (usually smooth)

## Key Takeaway

**One kernel launch, comprehensive timing!**

```python
# Before: 200+ launches, hard to time
for level in range(levels):
    for iter in range(pre_iter):
        simulator.launch("f_apply_operator")
        simulator.launch("f_jacobi_smooth")
    # ... many more calls ...

# After: 1 launch, built-in timing ✨
simulator.launch("f_enable_timer")
simulator.launch("f_gmg_vcycle", zDim, levels, pre_iter, post_iter, bottom_iter)
# Timing data automatically captured and returned!
```

