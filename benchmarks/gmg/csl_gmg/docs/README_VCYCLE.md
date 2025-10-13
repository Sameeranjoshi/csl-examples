# GMG V-Cycle State Machine Implementation

This directory now contains **two implementations** of the Geometric Multigrid solver:

## 1. Host-Controlled Version (Original)

**Files:**
- `src/kernel_gmg.csl` - Device kernel with individual functions
- `src/layout_gmg.csl` - Layout for host-controlled execution
- `run_gmg.py` - Host script that calls each operation separately

**Execution Pattern:**
```python
# Host explicitly calls each operation
for level in range(levels-1):
    simulator.launch("f_gmg_init", ...)
    for _ in range(pre_smooth_iter):
        simulator.launch("f_apply_operator", ...)
        simulator.launch("f_jacobi_smooth", ...)
    simulator.launch("f_apply_operator", ...)
    simulator.launch("f_residual", ...)
    simulator.launch("f_reduction_top_left_pattern", ...)
    simulator.launch("f_restriction_division", ...)
    # ... and so on
```

**Pros:**
- Easy to debug (can inspect intermediate results)
- Flexible (can modify flow from host)

**Cons:**
- High RPC launch latency overhead
- Multiple host-device synchronizations
- Slower performance

## 2. State Machine Version

**Files:**
- `src/kernel_gmg_vcycle.csl` - Device kernel with state machine
- `src/layout_gmg_vcycle.csl` - Layout for state machine execution
- `run_gmg_vcycle.py` - Simplified host script
- `commands_vcycle_wse3.sh` - WSE3 execution script

**Execution Pattern:**
```python
# Host makes ONE call, device manages everything
simulator.launch("f_gmg_vcycle", zDim, levels, pre_iter, post_iter, bottom_iter, nonblock=False)
```

## State Machine Design

The state machine implements the complete V-cycle:

### States:

**Initialization:**
- `STATE_INIT` - Start of V-cycle

**Down Cycle (for each level):**
- `STATE_DOWN_INIT` - Initialize level
- `STATE_DOWN_SMOOTH_APPLY` - Apply operator (A*u)
- `STATE_DOWN_SMOOTH_UPDATE` - Jacobi update
- `STATE_DOWN_SMOOTH_CHECK` - Check if more smoothing needed
- `STATE_DOWN_RESIDUAL_APPLY` - Apply operator for residual
- `STATE_DOWN_RESIDUAL_COMPUTE` - Compute r = f - Au
- `STATE_DOWN_RESTRICT_REDUCE` - Reduction to top-left
- `STATE_DOWN_RESTRICT_DIV` - Average restriction result
- `STATE_DOWN_LEVEL_CHECK` - Move to next coarser level

**Coarse Level:**
- `STATE_COARSE_INIT` - Initialize coarsest level
- `STATE_COARSE_SMOOTH_APPLY` - Apply operator
- `STATE_COARSE_SMOOTH_UPDATE` - Jacobi update
- `STATE_COARSE_SMOOTH_CHECK` - Check iterations

**Up Cycle (for each level):**
- `STATE_UP_INIT` - Initialize level
- `STATE_UP_BCAST` - Broadcast from top-left
- `STATE_UP_INTERP_ADD` - Add interpolated correction
- `STATE_UP_SMOOTH_APPLY` - Apply operator
- `STATE_UP_SMOOTH_UPDATE` - Jacobi update
- `STATE_UP_SMOOTH_CHECK` - Check if more smoothing needed
- `STATE_UP_LEVEL_CHECK` - Move to next finer level

**Completion:**
- `STATE_EXIT` - V-cycle complete

### Key Features:

1. **Asynchronous Operations**: Uses `f_trigger_state_machine()` as callback
2. **Level Tracking**: `current_level` tracks which multigrid level we're on
3. **Iteration Tracking**: `current_smooth_iter` tracks smoothing iterations
4. **Grid Spacing Arrays**: `hx_array`, `hy_array`, `hz_array` store spacing for all levels
5. **Jacobi Coefficients**: Pre-calculated for each level

