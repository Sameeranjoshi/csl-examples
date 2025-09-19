#!/usr/bin/env python3
"""
Geometric Multigrid Solver - Python Implementation
Modeled after the CUDA Bricks library implementation

This implementation replicates the same algorithm, parameters, and structure
as the CUDA GMG code for comparison experiments.

Key parameters from CUDA implementation:
- 7-point Poisson stencil: α = -6, β = 1
- Jacobi relaxation coefficient: 1/9 ≈ 0.111
- Pre-smooth iterations: 6
- Post-smooth iterations: 6
- Bottom solver iterations: 100
- Convergence tolerance: 1e-10
"""

import numpy as np
import time
from typing import Tuple, List, Optional


class GMGSolver:
    """Geometric Multigrid Solver for 3D Poisson equation"""
    
    def __init__(self, nx: int, ny: int, nz: int, num_levels: int = 6):
        """
        Initialize the GMG solver
        
        Args:
            nx, ny, nz: Grid dimensions
            num_levels: Number of multigrid levels
        """
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.num_levels = num_levels
        
        # Stencil coefficients (same as CUDA implementation)
        self.ALPHA = -6.0  # MPI_ALPHA
        self.BETA = 1.0    # MPI_BETA
        
        # Jacobi relaxation coefficient (same as CUDA implementation)
        self.JACOBI_COEFF = 1.0 / 9.0
        
        # Iteration counts (same as CUDA implementation)
        self.PRE_SMOOTH_ITER = 6
        self.POST_SMOOTH_ITER = 6
        self.BOTTOM_SOLVER_ITER = 100
        
        # Convergence tolerance
        self.TOLERANCE = 1e-10
        
        # Initialize grid hierarchy
        self.grids = self._create_grid_hierarchy()
        
        # Initialize right-hand side with sin function (same as CUDA)
        self._initialize_rhs()
        
        # Timing statistics
        self.timers = {
            'apply_op': 0.0,
            'smooth': 0.0,
            'restriction': 0.0,
            'interpolation': 0.0,
            'total': 0.0
        }
        
    def _create_grid_hierarchy(self) -> List[dict]:
        """Create the multigrid hierarchy"""
        grids = []
        
        for level in range(self.num_levels):
            # Calculate dimensions for this level
            nx = self.nx // (2 ** level)
            ny = self.ny // (2 ** level)
            nz = self.nz // (2 ** level)
            
            # Ensure minimum size and that we can do 2:1 coarsening
            nx = max(nx, 16)  # Need at least 16 to coarsen to 8
            ny = max(ny, 16)
            nz = max(nz, 16)
            
            
            # Create grid data
            grid = {
                'nx': nx,
                'ny': ny,
                'nz': nz,
                'x': np.zeros((nz, ny, nx), dtype=np.float64),  # Solution
                'rhs': np.zeros((nz, ny, nx), dtype=np.float64),  # Right-hand side
                'res': np.zeros((nz, ny, nx), dtype=np.float64),  # Residual
                'ax': np.zeros((nz, ny, nx), dtype=np.float64),   # A*x
                'h': 1.0 / (nx - 1)  # Grid spacing
            }
            
            grids.append(grid)
            
        return grids
    
    def _initialize_rhs(self):
        """Initialize right-hand side with sin function (same as CUDA implementation)"""
        for level, grid in enumerate(self.grids):
            nx, ny, nz = grid['nx'], grid['ny'], grid['nz']
            h = grid['h']
            
            # Create coordinate arrays
            x = np.linspace(0, 1, nx)
            y = np.linspace(0, 1, ny)
            z = np.linspace(0, 1, nz)
            X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
            
            # Initialize with sin function (same as prodSinArray in CUDA)
            grid['rhs'] = np.sin(2 * np.pi * X) * np.sin(2 * np.pi * Y) * np.sin(2 * np.pi * Z)
    
    def apply_operator(self, level: int) -> None:
        """Apply 7-point Poisson stencil operator (same as CUDA applyOp_kernel)"""
        grid = self.grids[level]
        x = grid['x']
        ax = grid['ax']
        
        nx, ny, nz = grid['nx'], grid['ny'], grid['nz']
        
        # Apply 7-point stencil: α*x[i,j,k] + β*(x[i±1,j,k] + x[i,j±1,k] + x[i,j,k±1])
        ax.fill(0.0)
        
        # Interior points only (same as CUDA with boundary checks)
        for k in range(1, nz-1):
            for j in range(1, ny-1):
                for i in range(1, nx-1):
                    ax[k, j, i] = (self.ALPHA * x[k, j, i] +
                                   self.BETA * (x[k, j, i+1] + x[k, j, i-1] +
                                               x[k, j+1, i] + x[k, j-1, i] +
                                               x[k+1, j, i] + x[k-1, j, i]))
    
    def jacobi_smooth(self, level: int, compute_residual: bool = False) -> None:
        """Jacobi smoother (same as CUDA smooth_kernel/smooth_residual_kernel)"""
        grid = self.grids[level]
        x = grid['x']
        rhs = grid['rhs']
        h = grid['h']
        
        # Apply operator
        self.apply_operator(level)
        ax = grid['ax']
        
        # Jacobi update: x' = x + ω * (rhs - Ax) / h²
        # where ω = JACOBI_COEFF and h² scaling is handled by dom_len_dev[0] in CUDA
        if compute_residual:
            # Compute residual: res = rhs - Ax
            grid['res'] = rhs - ax / (h * h)
        
        # Jacobi smoothing: x += JACOBI_COEFF * (Ax - h² * rhs)
        x += self.JACOBI_COEFF * (ax - (h * h) * rhs)
    
    def restriction(self, fine_level: int) -> None:
        """Full weighting restriction (same as CUDA restriction_kernel)"""
        fine_grid = self.grids[fine_level]
        coarse_grid = self.grids[fine_level + 1]
        
        fine_res = fine_grid['res']
        coarse_rhs = coarse_grid['rhs']
        
        # If grids are the same size, just copy
        if (fine_grid['nx'] == coarse_grid['nx'] and 
            fine_grid['ny'] == coarse_grid['ny'] and 
            fine_grid['nz'] == coarse_grid['nz']):
            coarse_rhs[:] = fine_res[:]
            return
        
        # Full weighting: average 8 fine points to 1 coarse point
        for k in range(coarse_grid['nz']):
            for j in range(coarse_grid['ny']):
                for i in range(coarse_grid['nx']):
                    # Map coarse indices to fine indices
                    fi = 2 * i
                    fj = 2 * j
                    fk = 2 * k
                    
                    # Check bounds
                    if (fi+1 < fine_grid['nx'] and fj+1 < fine_grid['ny'] and fk+1 < fine_grid['nz']):
                        # Average 8 fine points (same as CUDA implementation)
                        coarse_rhs[k, j, i] = (
                            fine_res[fk, fj, fi] + fine_res[fk, fj, fi+1] +
                            fine_res[fk, fj+1, fi] + fine_res[fk, fj+1, fi+1] +
                            fine_res[fk+1, fj, fi] + fine_res[fk+1, fj, fi+1] +
                            fine_res[fk+1, fj+1, fi] + fine_res[fk+1, fj+1, fi+1]
                        ) / 8.0
                    else:
                        # Handle boundary cases
                        coarse_rhs[k, j, i] = fine_res[fk, fj, fi]
    
    def interpolation(self, coarse_level: int) -> None:
        """Linear interpolation (same as CUDA interpolation_incr_kernel)"""
        coarse_grid = self.grids[coarse_level]
        fine_grid = self.grids[coarse_level - 1]
        
        coarse_x = coarse_grid['x']
        fine_x = fine_grid['x']
        
        # If grids are the same size, just copy
        if (fine_grid['nx'] == coarse_grid['nx'] and 
            fine_grid['ny'] == coarse_grid['ny'] and 
            fine_grid['nz'] == coarse_grid['nz']):
            fine_x[:] += coarse_x[:]
            return
        
        # Linear interpolation: copy coarse value to 8 fine points
        for k in range(coarse_grid['nz']):
            for j in range(coarse_grid['ny']):
                for i in range(coarse_grid['nx']):
                    # Map coarse indices to fine indices
                    fi = 2 * i
                    fj = 2 * j
                    fk = 2 * k
                    
                    # Check bounds and copy coarse value to fine points
                    coarse_val = coarse_x[k, j, i]
                    if (fi < fine_grid['nx'] and fj < fine_grid['ny'] and fk < fine_grid['nz']):
                        fine_x[fk, fj, fi] += coarse_val
                    if (fi+1 < fine_grid['nx'] and fj < fine_grid['ny'] and fk < fine_grid['nz']):
                        fine_x[fk, fj, fi+1] += coarse_val
                    if (fi < fine_grid['nx'] and fj+1 < fine_grid['ny'] and fk < fine_grid['nz']):
                        fine_x[fk, fj+1, fi] += coarse_val
                    if (fi+1 < fine_grid['nx'] and fj+1 < fine_grid['ny'] and fk < fine_grid['nz']):
                        fine_x[fk, fj+1, fi+1] += coarse_val
                    if (fi < fine_grid['nx'] and fj < fine_grid['ny'] and fk+1 < fine_grid['nz']):
                        fine_x[fk+1, fj, fi] += coarse_val
                    if (fi+1 < fine_grid['nx'] and fj < fine_grid['ny'] and fk+1 < fine_grid['nz']):
                        fine_x[fk+1, fj, fi+1] += coarse_val
                    if (fi < fine_grid['nx'] and fj+1 < fine_grid['ny'] and fk+1 < fine_grid['nz']):
                        fine_x[fk+1, fj+1, fi] += coarse_val
                    if (fi+1 < fine_grid['nx'] and fj+1 < fine_grid['ny'] and fk+1 < fine_grid['nz']):
                        fine_x[fk+1, fj+1, fi+1] += coarse_val
    
    def v_cycle(self, start_level: int = 0) -> None:
        """V-cycle multigrid (same as CUDA vcycle_brick)"""
        # Going down: pre-smooth and restrict
        for level in range(start_level, self.num_levels - 1):
            # Pre-smoothing
            for _ in range(self.PRE_SMOOTH_ITER):
                self.jacobi_smooth(level)
            
            # Compute residual for restriction
            self.jacobi_smooth(level, compute_residual=True)
            
            # Restriction
            self.restriction(level)
            
            # Initialize coarse solution
            self.grids[level + 1]['x'].fill(0.0)
        
        # Bottom solver
        bottom_level = self.num_levels - 1
        for _ in range(self.BOTTOM_SOLVER_ITER):
            self.jacobi_smooth(bottom_level)
        
        # Going up: interpolate and post-smooth
        for level in range(self.num_levels - 2, start_level - 1, -1):
            # Interpolation
            self.interpolation(level)
            
            # Post-smoothing
            for _ in range(self.POST_SMOOTH_ITER):
                self.jacobi_smooth(level)
    
    def compute_residual_norm(self, level: int = 0) -> float:
        """Compute maximum residual norm (same as CUDA maxNorm_brick)"""
        grid = self.grids[level]
        
        # Apply operator
        self.apply_operator(level)
        
        # Compute residual: res = rhs - Ax
        grid['res'] = grid['rhs'] - grid['ax'] / (grid['h'] * grid['h'])
        
        # Return maximum absolute residual
        return np.max(np.abs(grid['res']))
    
    def solve(self, max_iterations: int = 20) -> Tuple[float, int, dict]:
        """
        Solve the system using multigrid V-cycles
        
        Args:
            max_iterations: Maximum number of V-cycles
            
        Returns:
            Tuple of (final_residual, iterations, timing_stats)
        """
        # Initialize solution
        for grid in self.grids:
            grid['x'].fill(0.0)
        
        # Reset timers
        for key in self.timers:
            self.timers[key] = 0.0
        
        start_time = time.time()
        iterations = 0
        residual = 1.0
        
        print(f"Starting GMG solve with {max_iterations} max iterations")
        print(f"Grid size: {self.nx}x{self.ny}x{self.nz}, Levels: {self.num_levels}")
        print(f"Tolerance: {self.TOLERANCE}")
        print("-" * 50)
        
        while residual > self.TOLERANCE and iterations < max_iterations:
            # Perform V-cycle
            cycle_start = time.time()
            self.v_cycle()
            cycle_time = time.time() - cycle_start
            
            # Compute residual
            residual = self.compute_residual_norm()
            iterations += 1
            
            self.timers['total'] += cycle_time
            
            print(f"Iteration {iterations:2d}: Residual = {residual:.6e}, Time = {cycle_time:.4f}s")
        
        total_time = time.time() - start_time
        
        print("-" * 50)
        print(f"Converged after {iterations} iterations")
        print(f"Final residual: {residual:.6e}")
        print(f"Total time: {total_time:.4f}s")
        print(f"Time per iteration: {total_time/iterations:.4f}s")
        
        return residual, iterations, self.timers


def main():
    """Main function to run the GMG solver"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Geometric Multigrid Solver')
    parser.add_argument('-s', '--size', default='512,512,512', 
                       help='Grid size as nx,ny,nz (default: 512,512,512)')
    parser.add_argument('-l', '--levels', type=int, default=6,
                       help='Number of multigrid levels (default: 6)')
    parser.add_argument('-n', '--max_iter', type=int, default=20,
                       help='Maximum number of iterations (default: 20)')
    parser.add_argument('-I', '--iterations', type=int, default=1,
                       help='Number of times to run for timing (default: 1)')
    
    args = parser.parse_args()
    
    # Parse grid size
    size_parts = args.size.split(',')
    if len(size_parts) != 3:
        raise ValueError("Grid size must be in format nx,ny,nz")
    
    nx, ny, nz = map(int, size_parts)
    
    print("=" * 60)
    print("Geometric Multigrid Solver - Python Implementation")
    print("Modeled after CUDA Bricks library implementation")
    print("=" * 60)
    
    # Run solver multiple times for timing statistics
    times = []
    for run in range(args.iterations):
        print(f"\nRun {run + 1}/{args.iterations}")
        print("-" * 30)
        
        solver = GMGSolver(nx, ny, nz, args.levels)
        residual, iterations, timers = solver.solve(args.max_iter)
        
        times.append(timers['total'])
    
    # Print timing statistics
    if args.iterations > 1:
        times = np.array(times)
        print(f"\nTiming Statistics ({args.iterations} runs):")
        print(f"Mean time: {np.mean(times):.4f}s")
        print(f"Min time:  {np.min(times):.4f}s")
        print(f"Max time:  {np.max(times):.4f}s")
        print(f"Std dev:   {np.std(times):.4f}s")


if __name__ == "__main__":
    main()
