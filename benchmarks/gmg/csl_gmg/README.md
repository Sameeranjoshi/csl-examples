# Geometric Multigrid (GMG) Solver - CSL Implementation

This directory contains a CSL (Cerebras Software Language) implementation of a Geometric Multigrid solver for the 3D Poisson equation, designed to run on Cerebras WSE hardware.

## Overview

The implementation follows the same pattern as the conjugate gradient example, with host-side Python code coordinating device-side CSL kernels. The GMG algorithm implements:

- **Jacobi smoothing**: Iterative relaxation method
- **Restriction**: Full weighting from fine to coarse grid
- **Interpolation**: Linear interpolation from coarse to fine grid
- **7-point Poisson stencil**: Finite difference operator

## Key Features

- **CSL kernels**: Device-side implementation of GMG operations
- **Host coordination**: Python script manages data transfer and kernel launches
- **Modular design**: Separate kernels for each GMG operation
- **Timing support**: Built-in performance measurement
- **Memory efficient**: Optimized data layouts for WSE

## File Structure

```
csl_gmg/
├── src/
│   ├── kernel_gmg.csl      # Main GMG kernel implementation
│   ├── layout_gmg.csl      # Layout and compilation configuration
│   └── blas.csl            # Basic linear algebra operations
├── run_gmg.py              # Host-side Python script
├── commands_wse2.sh        # WSE2 execution script
├── commands_wse3.sh        # WSE3 execution script
└── README.md               # This file
```

## Algorithm Components

### 1. Jacobi Smoothing
```csl
fn f_jacobi_smooth(iterations: i16) void {
    // Apply operator: Au = A*u
    // Update: u_new = u + omega * (f - Au) / diagonal
}
```

### 2. Restriction
```csl
fn f_restrict() void {
    // Full weighting: average 8 fine points to 1 coarse point
    // Currently implements simple injection
}
```

### 3. Interpolation
```csl
fn f_interpolate() void {
    // Linear interpolation: copy coarse value to 8 fine points
    // Currently implements simple injection
}
```

### 4. Operator Application
```csl
fn f_apply_operator() void {
    // Apply 7-point Poisson stencil: Au = A*u
    stencil_mod.spmv(n, &stencil_coeff, &u, &Au);
}
```

## Data Structures

The implementation uses the following key data structures:

- `u`: Solution vector
- `f`: Right-hand side vector
- `r`: Residual vector
- `Au`: Operator applied to solution
- `u_smooth`: Smoothed solution
- `r_coarse`: Coarse residual
- `u_coarse`: Coarse solution
- `correction`: Correction from coarse level

## Parameters

- **Grid dimensions**: nx × ny × nz (default: 8×8×8)
- **Jacobi relaxation**: ω = 0.5
- **Stencil coefficients**: α = -6 (center), β = 1 (neighbors)
- **Grid spacing**: h = 1/(nx-1)
- **Smoothing iterations**: 3 (configurable)

## Usage

### Compilation and Execution

```bash
# For WSE2
./commands_wse2.sh

# For WSE3
./commands_wse3.sh
```

### Manual Execution

```bash
# Compile
cslc ./src/layout_gmg.csl --arch wse2 --fabric-dims=10,10 --fabric-offsets=1,1 \
--params=width:8,height:8,MAX_ZDIM:8 --params=BLOCK_SIZE:2 \
--params=C0_ID:0,C1_ID:1,C2_ID:2,C3_ID:3,C4_ID:4,C5_ID:5,C6_ID:6,C7_ID:7,C8_ID:8 \
-o=out --memcpy --channels=1 --width-west-buf=0 --width-east-buf=0

# Run
cs_python ./run_gmg.py -m=8 -n=8 -k=8 --latestlink out --channels=1 \
--width-west-buf=0 --width-east-buf=0 --zDim=8 --run-only
```

## Algorithm Flow

1. **Initialization**: Set up grid hierarchy and parameters
2. **Operator Application**: Compute Au = A*u
3. **Residual Computation**: Compute r = f - Au
4. **Jacobi Smoothing**: Apply relaxation iterations
5. **Restriction**: Transfer residual to coarse grid
6. **Interpolation**: Transfer correction back to fine grid
7. **Correction**: Add coarse correction to fine solution

## Current Limitations

- **Single level**: Currently implements only one level (no recursive V-cycle)
- **Simple restriction/interpolation**: Uses injection instead of full weighting/linear interpolation
- **No convergence checking**: Does not implement iterative convergence
- **Basic timing**: Limited performance measurement

## Future Enhancements

- **Multi-level V-cycle**: Implement recursive multigrid hierarchy
- **Full weighting restriction**: Complete 8-point averaging
- **Linear interpolation**: Proper coarse-to-fine transfer
- **Convergence monitoring**: Iterative residual checking
- **Performance optimization**: Advanced timing and profiling

## Dependencies

- CSL compiler (cslc)
- Cerebras SDK
- Python 3.7+
- NumPy
- Cerebras runtime libraries

## References

- Geometric Multigrid theory and implementation
- Cerebras CSL documentation
- Conjugate Gradient CSL implementation (reference)
- 3D Poisson equation finite difference methods
