# Geometric Multigrid Solver - Python Implementation

This is a Python implementation of a Geometric Multigrid (GMG) solver for the 3D Poisson equation, modeled after the CUDA Bricks library implementation found in the `baselines/bricklib_gmg` directory.

## Algorithm Details

This implementation replicates the exact same algorithm, parameters, and structure as the CUDA GMG code:

### Key Parameters (matching CUDA implementation)
- **7-point Poisson stencil**: α = -6, β = 1
- **Jacobi relaxation coefficient**: 1/9 ≈ 0.111
- **Pre-smooth iterations**: 6
- **Post-smooth iterations**: 6
- **Bottom solver iterations**: 100
- **Convergence tolerance**: 1e-10

### Multigrid Components
- **Operator**: 7-point finite difference stencil for 3D Poisson equation
- **Smoother**: Jacobi relaxation with coefficient 1/9
- **Restriction**: Full weighting (averages 8 fine points to 1 coarse point)
- **Interpolation**: Linear interpolation (copies coarse value to 8 fine points)
- **Cycle**: V-cycle multigrid

### Grid Hierarchy
- Automatically creates multigrid hierarchy with 2:1 coarsening
- Default: 6 levels (configurable)
- Minimum grid size: 8³ points per level

## Usage

### Basic Usage
```bash
python gmg_solver.py
```

### Command Line Options
```bash
python gmg_solver.py -s 256,256,256 -l 6 -n 20 -I 10
```

Options:
- `-s, --size`: Grid size as nx,ny,nz (default: 512,512,512)
- `-l, --levels`: Number of multigrid levels (default: 6)
- `-n, --max_iter`: Maximum number of iterations (default: 20)
- `-I, --iterations`: Number of times to run for timing statistics (default: 1)

### Example: Matching CUDA Benchmark
```bash
python gmg_solver.py -s 512,512,512 -I 10 -l 6 -n 20
```

This matches the CUDA benchmark command:
```bash
./cuda -s 512,512,512 -I 10 -l 6 -n 20
```

## Implementation Details

### File Structure
- `gmg_solver.py`: Main GMG solver implementation
- `README.md`: This documentation
- `requirements.txt`: Python dependencies

### Core Classes
- `GMGSolver`: Main solver class containing all multigrid operations

### Key Methods
- `apply_operator()`: Applies 7-point Poisson stencil
- `jacobi_smooth()`: Jacobi relaxation smoother
- `restriction()`: Full weighting restriction
- `interpolation()`: Linear interpolation
- `v_cycle()`: Complete V-cycle multigrid
- `solve()`: Main solver loop with convergence checking

### Right-Hand Side
The solver initializes the right-hand side with the same sin function as the CUDA implementation:
```
rhs = sin(2πx) * sin(2πy) * sin(2πz)
```

## Comparison with CUDA Implementation

This Python implementation is designed to be a direct comparison baseline for the CUDA Bricks library implementation. It uses:

1. **Identical algorithm parameters** (iteration counts, coefficients, tolerance)
2. **Same multigrid components** (operators, smoothers, transfer operators)
3. **Matching problem setup** (sin RHS, same boundary conditions)
4. **Similar output format** (iteration counts, residual norms, timing)

## Performance Notes

This Python implementation is intended for:
- **Algorithm validation**: Verify correctness of the multigrid algorithm
- **Performance comparison**: Compare with optimized CUDA implementation
- **Educational purposes**: Understand the GMG algorithm structure
- **Baseline testing**: Test modifications before implementing in CUDA

For production use, the CUDA implementation will be significantly faster due to:
- GPU parallelization
- Optimized data layouts (Bricks)
- Memory bandwidth optimization
- Reduced communication overhead

## Dependencies

- Python 3.7+
- NumPy
- No additional dependencies required

## Example Output

```
============================================================
Geometric Multigrid Solver - Python Implementation
Modeled after CUDA Bricks library implementation
============================================================

Starting GMG solve with 20 max iterations
Grid size: 512x512x512, Levels: 6
Tolerance: 1e-10
--------------------------------------------------
Iteration  1: Residual = 1.234567e-01, Time = 2.3456s
Iteration  2: Residual = 1.234567e-02, Time = 2.3456s
...
Iteration 15: Residual = 8.765432e-11, Time = 2.3456s
--------------------------------------------------
Converged after 15 iterations
Final residual: 8.765432e-11
Total time: 35.1840s
Time per iteration: 2.3456s
```
