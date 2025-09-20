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
    
    def __init__(self, nx: int, ny: int, nz: int, num_levels: int = 6, verbose: bool = False):
        """
        Initialize the GMG solver
        
        Args:
            nx, ny, nz: Grid dimensions
            num_levels: Number of multigrid levels
            verbose: Whether to print detailed level information
        """
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.num_levels = num_levels
        self.verbose = verbose
        
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
        self.number_of_operations = {
            'apply_op': 0.0,
            'smooth': 0.0,
            'restriction': 0.0,
            'interpolation': 0.0,
            'total': 0.0
        }
        
        # Print level information if verbose
        if self.verbose:
            self._print_level_info()
            self._print_initialization_info()
            self._print_level0_data()
        
    def _create_grid_hierarchy(self) -> List[dict]:
        """Create the multigrid hierarchy"""
        # Use a Python list to store the grid dictionaries for each level in the multigrid hierarchy.
        grids = []
        
        for level in range(self.num_levels):
            # Calculate dimensions for this level (same as CUDA implementation)
            # This is calculating the shapes at each level for the grid
            # newlevel shape =initial shape / factor^(level)
            nx = self.nx // (2 ** level)
            ny = self.ny // (2 ** level)
            nz = self.nz // (2 ** level)
            
            # Ensure minimum size of 1 (same as CUDA - no artificial minimum)
            # Having grid size of 1 is that too small?
            nx = max(nx, 1)
            ny = max(ny, 1)
            nz = max(nz, 1)
            
            
            # Create grid data
            grid = {
                'nx': nx,
                'ny': ny,
                'nz': nz,
                'x': np.zeros((nz, ny, nx), dtype=np.float64),  # Solution
                'rhs': np.zeros((nz, ny, nx), dtype=np.float64),  # Right-hand side
                'res': np.zeros((nz, ny, nx), dtype=np.float64),  # Residual
                'ax': np.zeros((nz, ny, nx), dtype=np.float64),   # A*x
                'h': 1.0 / max(nx - 1, 1)  # Grid spacing (handle nx=1 case)
            }
            grids.append(grid)
            
        return grids
    
    def _print_level_info(self):
        """Print detailed information about the multigrid hierarchy and data structures"""
        print("\nMultigrid Level Hierarchy:")
        print("=" * 50)
        for level, grid in enumerate(self.grids):
            print(f"Level {level:2d}: {grid['nx']:3d} x {grid['ny']:3d} x {grid['nz']:3d} "
                  f"(h = {grid['h']:.6f}, points = {grid['nx']*grid['ny']*grid['nz']:,})")
        print("=" * 50)
        
        print("\nData Structures at Each Level:")
        print("=" * 120)
        print(f"{'Level':<6} {'Grid Size':<12} {'x (solution)':<15} {'rhs (RHS)':<15} {'res (residual)':<15} {'Ax (operator)':<15} {'Stencil':<20}")
        print("-" * 120)
        
        for level, grid in enumerate(self.grids):
            nx, ny, nz = grid['nx'], grid['ny'], grid['nz']
            grid_size = f"{nx}×{ny}×{nz}"
            
            # Data structure shapes
            x_shape = f"({nx},{ny},{nz})"
            rhs_shape = f"({nx},{ny},{nz})"
            res_shape = f"({nx},{ny},{nz})"
            ax_shape = f"({nx},{ny},{nz})"
            
            # Stencil information
            if level == 0:
                stencil_info = "7-point Poisson"
            else:
                stencil_info = "7-point Poisson"
            
            print(f"{level:<6} {grid_size:<12} {x_shape:<15} {rhs_shape:<15} {res_shape:<15} {ax_shape:<15} {stencil_info:<20}")
        
        print("-" * 120)
        print("Legend:")
        print("  x     = Solution vector (unknown)")
        print("  rhs   = Right-hand side vector (source term)")
        print("  res   = Residual vector (rhs - Ax)")
        print("  Ax    = Operator applied to solution (A*x)")
        print("  Stencil = Finite difference stencil used")
        print("=" * 120)
        
        print("\nStencil Coefficients:")
        print("-" * 40)
        print(f"  α (center) = {self.ALPHA}")
        print(f"  β (neighbors) = {self.BETA}")
        print("  7-point stencil: α*x[i,j,k] + β*(x[i±1,j,k] + x[i,j±1,k] + x[i,j,k±1])")
        print("-" * 40)
        
        print("\nMemory Usage Summary:")
        print("-" * 50)
        total_points = 0
        for level, grid in enumerate(self.grids):
            points = grid['nx'] * grid['ny'] * grid['nz']
            total_points += points
            memory_mb = (points * 4 * 8) / (1024 * 1024)  # 4 arrays × 8 bytes per float64
            print(f"  Level {level}: {points:,} points × 4 arrays = {memory_mb:.2f} MB")
        total_memory = (total_points * 4 * 8) / (1024 * 1024)
        print(f"  Total: {total_points:,} points × 4 arrays = {total_memory:.2f} MB")
        print("-" * 50)
        print()
    
    def _print_level0_data(self):
        """Print representative data from Level 0 before V-cycle starts"""
        print("Level 0 Data Samples (before V-cycle):")
        print("=" * 60)
        
        grid = self.grids[0]
        nx, ny, nz = grid['nx'], grid['ny'], grid['nz']
        
        # Print a 3x3x3 sample from the center of the domain
        center_i, center_j, center_k = nx//2, ny//2, nz//2
        
        print(f"Sample from center region (i={center_i-1}:{center_i+2}, j={center_j-1}:{center_j+2}, k={center_k-1}:{center_k+2}):")
        print()
        
        # Print x (solution) - should be all zeros initially
        print("x (solution) - initial values:")
        for k in range(max(0, center_k-1), min(nz, center_k+2)):
            print(f"  k={k}:")
            for j in range(max(0, center_j-1), min(ny, center_j+2)):
                row_values = []
                for i in range(max(0, center_i-1), min(nx, center_i+2)):
                    row_values.append(f"{grid['x'][k,j,i]:8.4f}")
                print(f"    j={j}: [{' '.join(row_values)}]")
        
        print()
        
        # Print rhs (right-hand side) - sin function values
        print("rhs (right-hand side) - sin function values:")
        for k in range(max(0, center_k-1), min(nz, center_k+2)):
            print(f"  k={k}:")
            for j in range(max(0, center_j-1), min(ny, center_j+2)):
                row_values = []
                for i in range(max(0, center_i-1), min(nx, center_i+2)):
                    row_values.append(f"{grid['rhs'][k,j,i]:8.4f}")
                print(f"    j={j}: [{' '.join(row_values)}]")
        
        print()
        
        # Print res (residual) - should be all zeros initially
        print("res (residual) - initial values:")
        for k in range(max(0, center_k-1), min(nz, center_k+2)):
            print(f"  k={k}:")
            for j in range(max(0, center_j-1), min(ny, center_j+2)):
                row_values = []
                for i in range(max(0, center_i-1), min(nx, center_i+2)):
                    row_values.append(f"{grid['res'][k,j,i]:8.4f}")
                print(f"    j={j}: [{' '.join(row_values)}]")
        
        print()
        
        # Print Ax (operator result) - should be all zeros initially
        print("Ax (operator result) - initial values:")
        for k in range(max(0, center_k-1), min(nz, center_k+2)):
            print(f"  k={k}:")
            for j in range(max(0, center_j-1), min(ny, center_j+2)):
                row_values = []
                for i in range(max(0, center_i-1), min(nx, center_i+2)):
                    row_values.append(f"{grid['ax'][k,j,i]:8.4f}")
                print(f"    j={j}: [{' '.join(row_values)}]")
        
        print("=" * 60)
        print()
    
    def _print_initialization_info(self):
        """Print all initialization parameters and coefficients"""
        print("Initialization Parameters:")
        print("=" * 50)
        print(f"Grid dimensions: {self.nx} × {self.ny} × {self.nz}")
        print(f"Number of levels: {self.num_levels}")
        print(f"Grid spacing (h): {self.grids[0]['h']:.6f}")
        print()
        
        print("Stencil Coefficients:")
        print(f"  α (center coefficient): {self.ALPHA}")
        print(f"  β (neighbor coefficient): {self.BETA}")
        print(f"  7-point stencil: α*x[i,j,k] + β*(x[i±1,j,k] + x[i,j±1,k] + x[i,j,k±1])")
        print(f"  Grid spacing scaling: h² = {self.grids[0]['h']**2:.6f} (used in residual computation)")
        print()
        
        print("Relaxation Parameters:")
        print(f"  Jacobi coefficient: {self.JACOBI_COEFF:.6f}")
        print(f"  Pre-smooth iterations: {self.PRE_SMOOTH_ITER}")
        print(f"  Post-smooth iterations: {self.POST_SMOOTH_ITER}")
        print(f"  Bottom solver iterations: {self.BOTTOM_SOLVER_ITER}")
        print()
        
        print("Convergence Parameters:")
        print(f"  Tolerance: {self.TOLERANCE:.2e}")
        print()
        
        print("Right-hand Side Initialization:")
        print(f"  Function: sin(2πx) * sin(2πy) * sin(2πz)")
        print(f"  Domain: [0,1] × [0,1] × [0,1]")
        print(f"  Grid points: {self.nx} × {self.ny} × {self.nz}")
        print()
        
        print("Boundary Conditions:")
        print("  Dirichlet boundary conditions (u = 0 on boundary)")
        print("  Interior points only for stencil application")
        print("=" * 50)
        print()
    
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
    
    # TODO: check this
    def jacobi_smooth(self, level: int, compute_residual: bool = False) -> None:
        """Jacobi smoother (same as CUDA smooth_kernel/smooth_residual_kernel)"""
        grid = self.grids[level]
        x = grid['x']
        rhs = grid['rhs']
        h = grid['h']
        
        # Apply operator
        self.apply_operator(level)
        ax = grid['ax']
        
        # Jacobi update: x' = x + ω * (rhs - Ax/ h²)
        # where ω = JACOBI_COEFF and h² scaling is handled by dom_len_dev[0] in CUDA
        # CUDA: dom_len_dev[0] = h², dom_len_dev[1] = 1/h²
        h2 = h * h
        
        if compute_residual:
            # Compute residual: res = rhs - Ax/h² (same as CUDA: rhs - Ax*dom_len_dev[1])
            grid['res'] = rhs - ax / h2
        
        # Jacobi smoothing: x += JACOBI_COEFF * (Ax - h² * rhs) (same as CUDA)
        x += self.JACOBI_COEFF * (ax - h2 * rhs)
    
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
        for k in range(coarse_grid['nz']):  # depth
            for j in range(coarse_grid['ny']):  # column
                for i in range(coarse_grid['nx']):  # row
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
        if self.verbose:
            print("  V-Cycle: Going DOWN (pre-smooth + restrict)")
        
        # Going down: pre-smooth and restrict
        for level in range(start_level, self.num_levels - 1):
            if self.verbose:
                grid = self.grids[level]
                print(f"    Level {level}: {grid['nx']}x{grid['ny']}x{grid['nz']} - "
                      f"Pre-smooth ({self.PRE_SMOOTH_ITER} iter) + restrict")
            
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
        if self.verbose:
            grid = self.grids[bottom_level]
            print(f"    Level {bottom_level}: {grid['nx']}x{grid['ny']}x{grid['nz']} - "
                  f"Bottom solver ({self.BOTTOM_SOLVER_ITER} iter)")
        
        for _ in range(self.BOTTOM_SOLVER_ITER):
            self.jacobi_smooth(bottom_level)
        
        if self.verbose:
            print("  V-Cycle: Going UP (interpolate + post-smooth)")
        
        # Going up: interpolate and post-smooth
        for level in range(self.num_levels - 2, start_level - 1, -1):
            if self.verbose:
                grid = self.grids[level]
                print(f"    Level {level}: {grid['nx']}x{grid['ny']}x{grid['nz']} - "
                      f"Interpolate + post-smooth ({self.POST_SMOOTH_ITER} iter)")
            
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
        
        # Compute residual: res = rhs - Ax/h² (same as CUDA scaling)
        h2 = grid['h'] * grid['h']
        grid['res'] = grid['rhs'] - grid['ax'] / h2
        
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
            if self.verbose:
                print(f"\nIteration {iterations + 1}:")
            
            cycle_start = time.time()
            self.v_cycle()
            cycle_time = time.time() - cycle_start
            
            # Compute residual
            residual = self.compute_residual_norm()
            iterations += 1
            
            self.timers['total'] += cycle_time
            
            if self.verbose:
                print(f"  Final residual: {residual:.6e}, Time: {cycle_time:.4f}s")
            else:
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
    parser.add_argument('-s', '--size', default='16,16,16', 
                       help='Grid size as nx,ny,nz (nx = number of grid points in x (rows), ny = y (columns), nz = z (depth); default: 16,16,16)')
    parser.add_argument('-l', '--levels', type=int, default=3,
                       help='Number of multigrid levels (default: 3)')
    parser.add_argument('-n', '--max_iter', type=int, default=2,
                       help='Maximum number of iterations (default: 2)')
    parser.add_argument('-I', '--iterations', type=int, default=1,
                       help='Number of times to run for timing (default: 1)')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Print detailed level information')
    
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
        
        solver = GMGSolver(nx, ny, nz, args.levels, args.verbose)
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
