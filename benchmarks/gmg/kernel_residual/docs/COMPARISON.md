# GMG Implementation Comparison

## Quick Reference

### Original (Host-Controlled) vs State Machine (Device-Controlled)

| Feature | `run_gmg.py` | `run_gmg_vcycle.py` |
|---------|--------------|---------------------|
| **Kernel** | `kernel_gmg.csl` | `kernel_gmg_vcycle.csl` |
| **Layout** | `layout_gmg.csl` | `layout_gmg_vcycle.csl` |
| **State machine** | ❌ | ✅ |
| **RPC overhead** | High | **Minimal** |
| **Performance** | Slower | **Faster** |

## Code Comparison

### Host-Controlled Version (run_gmg.py)

```python
# Multiple simulator.launch() calls for each operation
for level in range(args.levels - 1):
    # Initialize
    simulator.launch("f_gmg_init", ...)
    
    # Jacobi smoothing (6 iterations)
    for i in range(PRE_SMOOTH_ITER):
        simulator.launch("f_apply_operator", ...)
        simulator.launch("f_jacobi_smooth", ...)
    
    # Residual
    simulator.launch("f_apply_operator", ...)
    simulator.launch("f_residual", ...)
    
    # Restriction
    simulator.launch("f_reduction_top_left_pattern", ...)
    simulator.launch("f_restriction_division", ...)
    
    # Copy data D2H, process, copy H2D
    residual_3d = copy_data_d2h_single(...)
    # ... more processing

# Coarse solve
simulator.launch("f_gmg_init", ...)
for i in range(BOTTOM_SOLVER_ITER):
    simulator.launch("f_apply_operator", ...)
    simulator.launch("f_jacobi_smooth", ...)

# Up cycle
for level in reversed(range(args.levels - 1)):
    simulator.launch("f_gmg_init", ...)
    simulator.launch("f_bcast_from_top_left", ...)
    simulator.launch("f_interpolation_add", ...)
    for i in range(POST_SMOOTH_ITER):
        simulator.launch("f_apply_operator", ...)
        simulator.launch("f_jacobi_smooth", ...)
```

**Total host-device roundtrips**: ~50-100+ depending on levels and iterations

### State Machine Version (run_gmg_vcycle.py)

```python
# Copy grid spacing and Jacobi coefficients once
simulator.memcpy_h2d(symbol_hx_array, hx_array, ...)
simulator.memcpy_h2d(symbol_hy_array, hy_array, ...)
simulator.memcpy_h2d(symbol_hz_array, hz_array, ...)
simulator.memcpy_h2d(symbol_jacobi_coeff_array, jacobi_coeff_array, ...)

# ONE call - entire V-cycle runs on device!
simulator.launch("f_gmg_vcycle", 
                np.int16(zDim), 
                np.int16(levels),
                np.int16(pre_iter),
                np.int16(post_iter),
                np.int16(bottom_iter),
                nonblock=False)

# Copy results back
u_wse_1d = np.zeros(...)
simulator.memcpy_d2h(u_wse_1d, symbol_u, ...)
```

**Total host-device roundtrips**: **1**

## When to Use Each Version

### Use Host-Controlled (`run_gmg.py`) when:
- ✅ Debugging algorithm
- ✅ Need to inspect intermediate values
- ✅ Experimenting with different algorithms
- ✅ Need flexibility to change flow

### Use State Machine (`run_gmg_vcycle.py`) when:
- ✅ Performance is critical
- ✅ Algorithm is stable and tested
- ✅ Want to minimize host-device communication
- ✅ Benchmarking or production use

## Implementation Details

### State Machine Flow

```
INIT
  ↓
DOWN_INIT (level 0)
  ↓
DOWN_SMOOTH (pre_iter times)
  ↓
DOWN_RESIDUAL
  ↓
DOWN_RESTRICT
  ↓
DOWN_INIT (level 1)
  ↓
... (repeat for all levels)
  ↓
COARSE_INIT (coarsest level)
  ↓
COARSE_SMOOTH (bottom_iter times)
  ↓
UP_INIT (level N-2)
  ↓
UP_BCAST
  ↓
UP_INTERP
  ↓
UP_SMOOTH (post_iter times)
  ↓
... (repeat for all levels)
  ↓
EXIT
```

### Key Data Passed to Device

1. **Grid spacing for all levels**: `hx_array[LEVELS]`, `hy_array[LEVELS]`, `hz_array[LEVELS]`
2. **Jacobi coefficients**: `jacobi_coeff_array[LEVELS]` (pre-calculated)
3. **Algorithm parameters**: `zDim`, `levels`, `pre_iter`, `post_iter`, `bottom_iter`

### Callbacks

Both `stencil_mod` and `reduce_mod` use `f_trigger_state_machine()` as their callback:
- When stencil operation completes → triggers next state
- When reduction completes → triggers next state

This ensures proper sequencing without host intervention.

## Performance Expectations


The speedup comes from:
1. Eliminating RPC launch latency (~100+ launches avoided)
2. No host-device synchronization overhead
3. Continuous device execution

