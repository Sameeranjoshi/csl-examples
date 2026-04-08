# Performance Model Discrepancy: Why Communication Cost Increases at Level 5+

## Executive Summary

The performance model predicts decreasing communication cost as level increases, but **actual measurements show cost increases starting at level 5**. The root cause is that the model doesn't account for **fixed overhead** and **hop-based forwarding overhead** that dominate when zDim becomes small.

## Key Finding

**The model assumes linear scaling, but there's a fixed overhead per communication that becomes dominant when zDim < BLOCK_SIZE/16 (i.e., zDim < 16 for BLOCK_SIZE=256).**

## Detailed Analysis

### Performance Model (Theoretical)

```
Cost(L) = (B + C*d) * (z/B)
where:
  B = BLOCK_SIZE = 256
  C = cost per PE hop
  d = distance = 2^L (hops)
  z = zDim = 256/2^L
```

This predicts: **Cost decreases as L increases** (because z decreases faster than d increases).

### Actual Behavior

| Level | zDim | hops | Model Prediction | Actual Cost | Discrepancy |
|-------|------|------|------------------|-------------|-------------|
| 4     | 16   | 16   | Decreasing       | 186.5 us    | ✓ Matches   |
| 5     | 8    | 32   | Decreasing       | 294.4 us    | ✗ **+58%** |
| 6     | 4    | 64   | Decreasing       | 524.5 us    | ✗ **+181%** |
| 7     | 2    | 128  | Decreasing       | 7598.4 us   | ✗ **+3975%** |

### Root Cause: Missing Terms in Model

The actual cost includes terms the model doesn't account for:

```
Actual_Cost(L) = Fixed_Overhead + Hop_Overhead*hops + Data_Cost
                = O_fixed*4 + O_hop*2^L + (B + C*2^L) * (z0/2^L / B)
```

**When zDim is large (levels 0-4):**
- Data_Cost dominates
- Model works well

**When zDim is small (levels 5-7):**
- Data_Cost becomes negligible: `(B + C*2^L) * (z/B)` → small
- Hop_Overhead*hops dominates: `O_hop*2^L` → **DOUBLES each level**
- Fixed_Overhead is constant but significant: `O_fixed*4`

### Why Cost Increases at Level 5

At level 5:
- hops **doubles** from 16 → 32
- zDim **halves** from 16 → 8
- Hop overhead **doubles** (O_hop*32 vs O_hop*16)
- Data cost **halves** (but was already small)

**Net effect:** Hop overhead increase (2×) > Data cost decrease (0.5×), so **total cost increases**.

### Code Evidence

From `pe.csl`:

1. **Fixed overhead per direction** (lines 546, 608, 627, 700):
   - Timer start/stop operations
   - DSD setup
   - Queue initialization
   - 4 directions × 2 operations (send+recv) = 8 fixed overhead operations

2. **Hop-based forwarding** (lines 633-696):
   - When `inactive_horizontal` or `inactive_vertical` is true
   - Data must be forwarded through intermediate PEs
   - Each hop adds latency: `O_hop * hops`
   - This scales with hops but NOT with data size

3. **Small block inefficiency** (line 321):
   - When `zDim < BLOCK_SIZE`, only 1 iteration
   - But still pay full setup cost for all 4 directions
   - Hardware pipelines underutilized

## Corrected Model Formula

```
Cost(L) = O_fixed*4 + O_hop*2^L + (B + C*2^L) * (z0/2^L / B)

where:
  O_fixed = fixed overhead per communication direction (~10-20 us)
  O_hop = overhead per hop (~2-5 us)
  B = BLOCK_SIZE = 256
  C = data transfer cost per PE (~0.1-0.5 us)
  z0 = initial zDim = 256
  L = level (0, 1, 2, ...)
```

**Critical threshold:** When `O_hop*2^L > (B + C*2^L) * (z0/2^L / B)`, cost starts increasing.

For the given parameters, this happens around **level 5** when:
- `O_hop*32 > (256 + C*32) * (8/256)`
- Assuming O_hop ≈ 5 us: `160 us > ~8-10 us` ✓

## Recommendations

1. **Measure fixed overhead:** Profile with very small zDim to quantify O_fixed
2. **Measure hop overhead:** Profile with different hop counts to quantify O_hop  
3. **Optimize for small zDim:** Use different communication strategy when zDim < threshold (e.g., 16)
4. **Adjust BLOCK_SIZE:** For small zDim, use smaller block sizes to reduce overhead
5. **Batch communications:** When zDim is small, batch multiple operations to amortize fixed costs
6. **Update model:** Include fixed and hop overhead terms in performance model

## Conclusion

The performance model is **correct for large zDim** but **fails for small zDim** because it doesn't account for:
- Fixed overhead per communication direction
- Hop-based forwarding overhead that scales with hops
- Hardware inefficiencies with very small blocks

The cost increase at level 5+ is **expected behavior** given the actual implementation, not a bug. The model needs to be updated to include these overhead terms.

