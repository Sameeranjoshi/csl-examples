# Performance Model Analysis: Communication Cost Anomaly

## Performance Model (Theoretical)

Given:
- **B** = BLOCK_SIZE (256 in this case)
- **C** = cost of passing one PE
- **z** = zDim at level L = z0/2^L (where z0 = 256)
- **d** = distance between active PEs = 2^L
- **hops** = factor = 2^L (from code: `hops = factor = 2^level_id`)

**Cost at level L:**
```
Cost(L) = (B + C*d) * (z/B)
       = (B + C*2^L) * (z0/2^L / B)
       = (B + C*2^L) * (z0/(B*2^L))
```

**Cost at level L+1:**
```
Cost(L+1) = (B + C*2^(L+1)) * (z0/2^(L+1) / B)
          = (B + C*2*2^L) * (z0/(B*2*2^L))
          = (B/2 + C*2^L) * (z0/(B*2^L))
```

**Difference:**
```
Cost(L) - Cost(L+1) = (B/2) * (z0/(B*2^L)) = z0/2^(L+1)
```

This should **decrease** as L increases.

## Actual Measurements (from response.txt)

| Level | zDim | hops | Communication Time (us) | Expected Trend |
|-------|------|------|--------------------------|----------------|
| 0     | 256  | 1    | 491.049                  | Baseline       |
| 1     | 128  | 2    | 331.371                  | ↓ Decreasing   |
| 2     | 64   | 4    | 272.599                  | ↓ Decreasing   |
| 3     | 32   | 8    | 250.117                  | ↓ Decreasing   |
| 4     | 16   | 16   | 186.507                  | ↓ Decreasing   |
| 5     | 8    | 32   | 294.365                  | ⚠️ **INCREASING** |
| 6     | 4    | 64   | 524.531                  | ⚠️ **INCREASING** |
| 7     | 2    | 128  | 7598.425                 | ⚠️ **HUGE INCREASE** |

## Root Cause Analysis

### Issue 1: Fixed Overhead Per Communication

The performance model assumes **linear scaling**, but there's likely a **fixed overhead** per communication operation that becomes dominant when zDim is small.

When `zDim < BLOCK_SIZE`:
- Number of iterations = 1 (single block)
- But we still have **4 communication directions** (east, west, north, south)
- Each direction has setup overhead that doesn't scale with zDim

**Actual cost might be:**
```
Cost(L) = (Fixed_Overhead + (B + C*2^L) * (z/B)) * Num_Directions
```

When zDim becomes very small (zDim << B), the fixed overhead dominates:
- Level 4: zDim=16, overhead might be ~10-20% of total
- Level 5: zDim=8, overhead becomes ~50%+ of total
- Level 6: zDim=4, overhead dominates
- Level 7: zDim=2, overhead completely dominates

### Issue 2: Hop-Based Forwarding Overhead

From the code (`pe.csl` lines 633-696), when `inactive_horizontal` or `inactive_vertical` is true, PEs forward data:
- This adds **extra communication steps** per hop
- The overhead per hop might be significant
- As hops increases (32, 64, 128), this overhead accumulates

**At level 5:**
- hops = 32 (vs 16 at level 4)
- More intermediate PEs involved in forwarding
- Each forwarding operation has overhead

### Issue 3: Small zDim Inefficiency

When zDim is very small:
- **Underutilization**: BLOCK_SIZE=256 but zDim=8, so we're only using 8/256 = 3.125% of the block
- **Pipeline stalls**: Hardware pipelines might not be fully utilized
- **DSD setup overhead**: Multiple DSD operations for small data might have fixed costs

### Issue 4: Hardware-Specific Behavior

The Cerebras hardware might have:
- **Minimum communication latency** that doesn't scale down with data size
- **Queue setup/teardown costs** that are fixed per communication
- **Routing overhead** that increases with hop count

## Corrected Performance Model

The actual cost should account for:

1. **Fixed overhead per communication direction:**
   ```
   Cost_fixed = O_setup * Num_Directions
   ```
   - Each direction (east, west, north, south) has setup overhead
   - Timer start/stop operations (lines 546, 608, 627, 700 in pe.csl)
   - DSD setup and queue initialization

2. **Variable cost per hop:**
   ```
   Cost_per_hop = O_hop * hops
   ```
   - Each hop adds latency
   - Forwarding through inactive PEs adds overhead (lines 633-696)
   - Routing overhead increases with hop count

3. **Data transfer cost:**
   ```
   Cost_data = (B + C*2^L) * (z/B)  [original model]
   ```
   - This scales down as zDim decreases
   - But becomes negligible when zDim << B

4. **Total cost:**
   ```
   Cost(L) = Cost_fixed + Cost_per_hop + Cost_data
           = O_setup*4 + O_hop*2^L + (B + C*2^L) * (z0/2^L / B)
   ```

When zDim is small (zDim << B), `Cost_data` becomes negligible, and fixed costs dominate:
```
Cost(L) ≈ O_setup*4 + O_hop*2^L  [when zDim is very small]
```

This explains why cost **increases** at level 5+ when hops becomes large (32, 64, 128).

## Code-Specific Issues

### Issue A: Sequential Communication Pattern
The code uses a **state machine** that processes directions sequentially:
- SEND_STATE_EAST → SEND_STATE_WEST → SEND_STATE_NORTH → SEND_STATE_SOUTH
- Each state transition has overhead
- Cannot pipeline or parallelize the 4 directions

### Issue B: Timer Overhead Per Direction
Looking at lines 546, 608, 627, 700:
- Timer is started at the beginning of each send/receive operation
- Timer is stopped at the end
- With 4 directions × 2 operations (send+recv) = 8 timer operations
- This overhead is **fixed** regardless of zDim size

### Issue C: Forwarding Through Inactive PEs
When `inactive_horizontal` or `inactive_vertical` is true (lines 633-696):
- Data must be forwarded through intermediate PEs
- Each forwarding operation has overhead
- As hops increases (32, 64, 128), more forwarding operations occur
- This creates a **multiplicative overhead** that the model doesn't account for

### Issue D: Small Block Inefficiency
When `zDim < BLOCK_SIZE`:
- `cur_length = zDim` (line 321)
- Only 1 iteration needed
- But we still pay full setup cost for all 4 directions
- Hardware pipelines might be underutilized with such small blocks

## Recommendations

1. **Measure fixed overhead**: Profile communication with very small zDim to quantify O_setup
2. **Measure hop overhead**: Profile with different hop counts to quantify O_hop
3. **Optimize for small zDim**: Consider different communication strategies when zDim < threshold
4. **Adjust BLOCK_SIZE**: For small zDim, use smaller block sizes to reduce overhead
5. **Batch communications**: When zDim is small, batch multiple operations to amortize fixed costs

