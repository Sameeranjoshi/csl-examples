# GMG V-Cycle Timing Measurements

## Overview

The GMG V-cycle state machine now includes comprehensive timing measurements for all major operations, similar to the CUDA bricks implementation.

## Measured Operations

### 1. **Smooth** (Pre-smooth + Post-smooth)
- **Down cycle**: Jacobi smoothing iterations before restriction
- **Up cycle**: Jacobi smoothing iterations after interpolation
- **Coarse level**: Bottom solver iterations
- **Combined**: Both down and up smoothing times are measured separately per level

### 2. **Apply Operator** (Residual computation)
- Application of 7-point Poisson stencil for residual calculation
- Measured only once per level in down cycle (after smoothing)

### 3. **Restriction**
- Reduction to top-left PE of restriction window
- Division by number of active PEs (averaging)
- Only measured at levels 0 to (levels-2)

### 4. **Interpolation**
- Broadcast from top-left PE
- Addition of coarse correction to fine solution
- Only measured at levels 0 to (levels-2) in up cycle

## Implementation Details

### CSL Kernel Side

#### Timing Buffers

```csl
// Per-operation, per-level timing (6 u16 per level = start[3] + end[3])
var timing_smooth = @zeros([LEVELS * 6]u16);
var timing_apply_op = @zeros([LEVELS * 6]u16);
var timing_restrict = @zeros([LEVELS * 6]u16);
var timing_interp = @zeros([LEVELS * 6]u16];
```

#### Capture Points

**Down Cycle Smoothing:**
```csl
STATE_DOWN_SMOOTH_APPLY (iter==0):
    save_timestamp_start(current_level, &timing_smooth)
    
STATE_DOWN_SMOOTH_CHECK (after all iters):
    save_timestamp_end(current_level, &timing_smooth)
```

**Residual Operator:**
```csl
STATE_DOWN_RESIDUAL_APPLY:
    save_timestamp_start(current_level, &timing_apply_op)
    
STATE_DOWN_RESIDUAL_COMPUTE:
    save_timestamp_end(current_level, &timing_apply_op)
```

**Restriction:**
```csl
STATE_DOWN_RESTRICT_REDUCE:
    save_timestamp_start(current_level, &timing_restrict)
    
STATE_DOWN_RESTRICT_DIV:
    save_timestamp_end(current_level, &timing_restrict)
```

**Interpolation:**
```csl
STATE_UP_BCAST:
    save_timestamp_start(current_level, &timing_interp)
    
STATE_UP_INTERP_ADD:
    save_timestamp_end(current_level, &timing_interp)
```

**Up Cycle Smoothing:**
```csl
STATE_UP_SMOOTH_APPLY (iter==0):
    save_timestamp_start(current_level, &timing_smooth)
    
STATE_UP_SMOOTH_CHECK (after all iters):
    save_timestamp_end(current_level, &timing_smooth)
```

### Python Host Side

#### Timing Data Retrieval

```python
# Copy timing arrays from device (each is LEVELS * 6 u16 values per PE)
timing_smooth_hwl = copy_timing_data(height, width, levels, simulator, symbol_timing_smooth)
timing_apply_op_hwl = copy_timing_data(height, width, levels, simulator, symbol_timing_apply_op)
timing_restrict_hwl = copy_timing_data(height, width, levels, simulator, symbol_timing_restrict)
timing_interp_hwl = copy_timing_data(height, width, levels, simulator, symbol_timing_interp)
```

#### Processing

For each operation at each level:
1. Extract start/end timestamps from all PEs
2. Calculate elapsed cycles: `cycles = time_end - time_start`
3. Compute min/max/avg across all PEs
4. Convert to microseconds: `time_us = (cycles / 0.85) * 1e-3`

## Output Format

### Per-Level Breakdown

```
Time per operation and level [min, avg, max] cycles (time_us):
--------------------------------------------------------------------------------
Level 0:
  smooth:        [    1234,     1250,     1267]  (   1.471 us)
  apply_op:      [     456,      460,      465]  (   0.541 us)
  restriction:   [     789,      800,      812]  (   0.941 us)
  interpolation: [     234,      240,      245]  (   0.282 us)

Level 1:
  smooth:        [     567,      580,      592]  (   0.682 us)
  apply_op:      [     123,      128,      132]  (   0.151 us)
  restriction:   [     345,      350,      356]  (   0.412 us)
  interpolation: [     112,      118,      123]  (   0.139 us)

Level 2 (coarse):
  smooth:        [   12345,    12560,    12780]  (  14.776 us)
  apply_op:      [      89,       95,      101]  (   0.112 us)
```

### Totals

```
================================================================================
Total Time Summary:
  Total smooth:           17.929 us
  Total apply_op:          0.804 us
  Total restriction:       1.353 us
  Total interpolation:     0.421 us
  Total V-cycle time:     20.507 us
================================================================================
```

## Comparison with Bricks CUDA

The timing measurements map to bricks timers as follows:

| CSL Measurement | Bricks Timer | Notes |
|-----------------|--------------|-------|
| `timing_smooth` | `timers.pr` | Smooth + residual combined in CSL |
| `timing_apply_op` | `timers.apply_op` | Operator application for residual |
| `timing_restrict` | `timers.restriction` | Full restriction operation |
| `timing_interp` | `timers.interpolation_incr` | Interpolation + increment |
| *(not measured)* | `timers.exchange_total` | No MPI exchange in single-device CSL |

## Important Notes

### 1. Smooth Timing Accumulation

**Important**: For levels that appear in both down and up cycles (levels 0 to levels-2), the `timing_smooth` captures BOTH pre-smoothing and post-smoothing times because we overwrite the same timing slots. 

**Down cycle**: First capture at level L
**Up cycle**: Second capture at level L (overwrites down cycle timing)

To get total smoothing time, you'd need to capture them separately. Currently, the timing shows the MOST RECENT smoothing operation.

**Fix option 1**: Use separate arrays for down/up smoothing
**Fix option 2**: Accumulate times instead of overwriting

### 2. Clock Speed

Cerebras WSE runs at **850 MHz**, so:
- 1 cycle = (1/0.85) ns = 1.176 ns
- 1000 cycles = 1.176 microseconds
- Conversion formula: `time_us = (cycles / 0.85) * 1e-3`

### 3. Multi-PE Timing

Each PE captures its own timestamps. The output shows:
- **Min**: Fastest PE (best case)
- **Max**: Slowest PE (worst case, usually determines overall time)
- **Avg**: Average across all PEs

### 4. Active vs Inactive PEs

At coarser levels, only a subset of PEs are active (e.g., level 2 with factor=4 has only 4 active PEs out of 64 total). The timing processing filters out zeros from inactive PEs.

## Usage

### Enable Timing

Timing is enabled automatically in the run script:

```python
simulator.launch("f_enable_timer", nonblock=False)
```

### Run V-Cycle with Timing

```bash
cs_python run_gmg_vcycle.py -m=8 -n=8 -k=8 --zDim=8 --levels=3 --max-ite=1
```

### Interpret Results

Look for the "Performance Timing" section in the output. Key metrics:
- **Smooth time**: Should dominate (most iterations)
- **Apply op time**: Should be moderate (stencil operations)
- **Restriction time**: Should be small (only reduction)
- **Interpolation time**: Should be small (only broadcast + add)

### Performance Optimization

If timing shows bottlenecks:
- **Smooth too slow**: Reduce iterations or optimize Jacobi coefficient calculation
- **Apply op too slow**: Optimize stencil communication pattern
- **Restriction too slow**: Check reduction tree efficiency
- **Interpolation too slow**: Check broadcast pattern

## Example Output

```
============================================================
Performance Timing (similar to bricks output)
============================================================

Time per operation and level [min, avg, max] cycles (time_us):
--------------------------------------------------------------------------------
Level 0:
  smooth:        [    5234,     5250,     5289]  (   6.176 us)
  apply_op:      [     892,      901,      915]  (   1.060 us)
  restriction:   [    1123,     1145,     1167]  (   1.347 us)
  interpolation: [     456,      465,      478]  (   0.547 us)

Level 1:
  smooth:        [    2345,     2367,     2389]  (   2.785 us)
  apply_op:      [     234,      245,      256]  (   0.288 us)
  restriction:   [     678,      689,      701]  (   0.811 us)
  interpolation: [     234,      241,      249]  (   0.284 us)

Level 2 (coarse):
  smooth:        [   23456,    23678,    23901]  (  27.856 us)
  apply_op:      [     178,      189,      201]  (   0.222 us)

================================================================================
Total Time Summary:
  Total smooth:           36.817 us
  Total apply_op:          1.570 us
  Total restriction:       2.158 us
  Total interpolation:     0.831 us
  Total V-cycle time:     41.376 us
================================================================================
```

## Future Enhancements

1. **Separate down/up smoothing times**: Track pre-smooth and post-smooth separately
2. **Per-iteration timing**: Measure each smoothing iteration individually
3. **State transition overhead**: Measure time spent in state transitions
4. **Memory access counters**: Use performance counters to track memory bandwidth
5. **Communication timing**: Measure stencil communication vs computation

