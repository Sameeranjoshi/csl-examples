#!/usr/bin/env python3
"""
Simplest Geometric Multigrid Solver for 3D Poisson equation
Pure CPU implementation without any complex considerations

Run: simplest example
python3 gmg.py -s 16,16,16 -l 3 -n 10 --tolerance 1e-3 -v
"""

import numpy as np
import time
from typing import Tuple, List

DTYPE = np.float32

class SimpleGMG:
    """Simplest possible GMG solver for 3D Poisson equation"""
    
    def __init__(self, nx: int, ny: int, nz: int, num_levels: int = 4, 
                verbose: bool = False, tolerance: float = 1e-6, 
                pre_iter: int = 6, post_iter: int = 6, bottom_iter: int = 100):
        """
        Initialize the GMG solver
        
        Args:
            nx, ny, nz: Grid dimensions
            num_levels: Number of multigrid levels
            verbose: Whether to print detailed level information
            tolerance: Convergence tolerance
            pre_iter: Number of pre-smoothing iterations
            post_iter: Number of post-smoothing iterations
            bottom_iter: Number of bottom solver iterations
        """
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.num_levels = num_levels
        self.verbose = verbose
        
        # Simple parameters
        self.omega = 1.0 / 2.0  # Jacobi relaxation parameter
        self.tolerance = tolerance
        self.ALPHA = -6.0
        self.BETA = 1.0

        # Iteration counts
        self.PRE_SMOOTH_ITER = pre_iter
        self.POST_SMOOTH_ITER = post_iter
        self.BOTTOM_SOLVER_ITER = bottom_iter
        
        # Create grid hierarchy
        self.grids = self._create_grids()
        
        # Initialize RHS
        self._init_rhs()
        
        # Print level information if verbose
        if self.verbose:
            self._print_level_info()
            self._print_initialization_info()
        
    def _create_grids(self):
        """Create simple grid hierarchy with semicoarsening (z dimension preserved)"""
        grids = []
        
        for level in range(self.num_levels):
            # Calculate dimensions for this level
            # Semicoarsening: only reduce x and y dimensions, keep z dimension
            nx = max(1, self.nx // (2 ** level))
            ny = max(1, self.ny // (2 ** level))
            nz = self.nz  # Keep z dimension unchanged for semicoarsening
            
            # Grid spacing
            h = 1.0 / max(nx - 1, 1)
            
            grid = {
                'nx': nx, 'ny': ny, 'nz': nz, 'h': h,
                'u': np.zeros((nx, ny, nz), dtype=DTYPE),      # Solution
                'f': np.zeros((nx, ny, nz), dtype=DTYPE),      # Right-hand side
                'r': np.zeros((nx, ny, nz), dtype=DTYPE),      # Residual
                'Au': np.zeros((nx, ny, nz), dtype=DTYPE)      # A*u
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
        print(f"{'Level':<6} {'Grid Size (nx×ny×nz)':<22} {'u shape (nx,ny,nz)':<22} "
            f"{'f shape':<15} {'r shape':<15} {'Au shape':<15} {'Stencil':<20}")
        print("-" * 120)

        for level, grid in enumerate(self.grids):
            nx, ny, nz = grid['nx'], grid['ny'], grid['nz']
            grid_size = f"{nx}×{ny}×{nz}"
            u_shape  = str(grid['u'].shape)
            f_shape  = str(grid['f'].shape)
            r_shape  = str(grid['r'].shape)
            au_shape = str(grid['Au'].shape)
            stencil_info = "7-point Poisson"
            print(f"{level:<6} {grid_size:<22} {u_shape:<22} {f_shape:<15} "
                f"{r_shape:<15} {au_shape:<15} {stencil_info:<20}")

        print("-" * 120)
        print("Legend:")
        print("  Arrays are stored as (nx, ny, nz) = (x, y, z)")
        print("-" * 120)
        print("Legend:")
        print("  u     = Solution vector (unknown)")
        print("  f     = Right-hand side vector (source term)")
        print("  r     = Residual vector (f - Au)")
        print("  Au    = Operator applied to solution (A*u)")
        print("  Stencil = Finite difference stencil used")
        print("=" * 120)
        
        print("\nStencil Coefficients:")
        print("-" * 40)
        print(f"  α (center) = {self.ALPHA}")
        print(f"  β (neighbors) = {self.BETA}")
        print("  7-point stencil: α*u[i,j,k] + β*(u[i±1,j,k] + u[i,j±1,k] + u[i,j,k±1])")
        print("-" * 40)
        
        print("\nMemory Usage Summary:")
        print("-" * 50)
        total_points = 0
        for level, grid in enumerate(self.grids):
            points = grid['nx'] * grid['ny'] * grid['nz']
            total_points += points
            bytes_per_elem = self.grids[0]['u'].dtype.itemsize  # f32
            memory_mb = (points * 4 * bytes_per_elem) / (1024 * 1024)
            print(f"  Level {level}: {points:,} points × 4 arrays = {memory_mb:.2f} MB")
        total_bytes_per_elem = self.grids[0]['u'].dtype.itemsize
        total_memory = (total_points * 4 * total_bytes_per_elem) / (1024 * 1024)
        print(f"  Total: {total_points:,} points × 4 arrays = {total_memory:.2f} MB")
        print("-" * 50)
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
        print(f"  7-point stencil: α*u[i,j,k] + β*(u[i±1,j,k] + u[i,j±1,k] + u[i,j,k±1])")
        print(f"  Grid spacing scaling: h² = {self.grids[0]['h']**2:.6f} (used in residual computation)")
        print()
        
        print("Relaxation Parameters:")
        print(f"  Jacobi coefficient: {self.omega:.6f}")
        print(f"  Pre-smooth iterations: {self.PRE_SMOOTH_ITER}")
        print(f"  Post-smooth iterations: {self.POST_SMOOTH_ITER}")
        print(f"  Bottom solver iterations: {self.BOTTOM_SOLVER_ITER}")
        print()
        
        print("Convergence Parameters:")
        print(f"  Tolerance: {self.tolerance:.2e}")
        print()
        
        print("Right-hand Side Initialization:")
        print(f"  Function: sin(πx) * sin(πy) * sin(πz)")
        print(f"  Domain: [0,1] × [0,1] × [0,1]")
        print(f"  Grid points: {self.nx} × {self.ny} × {self.nz}")
        print()
        
        print("Boundary Conditions:")
        print("  Dirichlet boundary conditions (u = 0 on boundary)")
        print("  Interior points only for stencil application")

        print("Initial values:")
        print("  x (solution) - 0.0:")
        print("  f (RHS) - sin function values:")
        print("  r (residual) - 0.0:")
        print("  Au (operator) - 0.0:")
        print()

        print("=" * 50)
        print()
    
    def _init_rhs(self):
        """Initialize right-hand side with simple test function"""
        for level, grid in enumerate(self.grids):
            nx, ny, nz = grid['nx'], grid['ny'], grid['nz']
            h = grid['h']
            
            # Create coordinate arrays
            x = np.linspace(0, 1, nx, dtype=DTYPE)
            y = np.linspace(0, 1, ny, dtype=DTYPE)
            z = np.linspace(0, 1, nz, dtype=DTYPE)
            X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
            
            # Simple test function: f = sin(πx)sin(πy)sin(πz)
            pi = DTYPE(np.pi)
            grid['f'] = np.sin(pi * X) * np.sin(pi * Y) * np.sin(pi * Z)
    
    def apply_operator(self, level: int):
        """Apply 7-point Laplacian operator"""
        grid = self.grids[level]
        u = grid['u']
        Au = grid['Au']
        h = grid['h']
        nx, ny, nz = grid['nx'], grid['ny'], grid['nz']
        alpha = DTYPE(self.ALPHA)
        beta = DTYPE(self.BETA)
        
        # Clear Au
        Au.fill(0.0)

        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    val = alpha * u[i, j, k]

                    # west/east
                    if i > 0:      val += beta * u[i-1, j, k]
                    if i < nx-1:   val += beta * u[i+1, j, k]

                    # south/north
                    if j > 0:      val += beta * u[i, j-1, k]
                    if j < ny-1:   val += beta * u[i, j+1, k]

                    # bottom/top
                    if k > 0:      val += beta * u[i, j, k-1]
                    if k < nz-1:   val += beta * u[i, j, k+1]
                    Au[i, j, k] = val / (h*h)


    def compute_residual(self, level: int):
        """Compute residual r = f - Au"""
        grid = self.grids[level]
        f = grid['f']
        r = grid['r']
        
        # Apply operator
        self.apply_operator(level)
        Au = grid['Au']
        
        # Residual: r = f - Au
        r[:] = f - Au
    
    def jacobi_smooth(self, level: int, num_iter: int = 1):
        """Jacobi smoothing"""
        grid = self.grids[level]
        u = grid['u']
        f = grid['f']
        h = grid['h']
        nx, ny, nz = grid['nx'], grid['ny'], grid['nz']
        omega = DTYPE(self.omega)
        
        for _ in range(num_iter):
            # Apply operator
            self.apply_operator(level)
            Au = grid['Au']
            
            # Jacobi update: u_new = u + ω * (f - Au) / diagonal
            # For 7-point stencil, diagonal is -6/h²
            diagonal = DTYPE(-6.0) / (h * h)
            update = omega * (f - Au) / diagonal
            
            # Update interior points only
            # u[1:-1, 1:-1, 1:-1] += update[1:-1, 1:-1, 1:-1]
            u[:] += update[:]  # update all cells, boundary handled by apply_operator
    
    def restrict(self, fine_level: int):
        """Semicoarsening restriction - only reduce in 2D (x,y), keep z dimension"""
        fine = self.grids[fine_level]
        coarse = self.grids[fine_level + 1]
        
        fine_r = fine['r'] # (nx, ny, nz)
        coarse_f = coarse['f']
        
        # If same size, just copy
        if (fine['nx'] == coarse['nx'] and 
            fine['ny'] == coarse['ny'] and 
            fine['nz'] == coarse['nz']):
            coarse_f[:] = fine_r[:]
            return
        
        # Semicoarsening: average 4 fine points in 2D to 1 coarse point
        # Keep z dimension unchanged (no reduction in z)
        # Simple: reduce over x and y for all z dimensions
        Xdim, Ydim, Zdim = coarse_f.shape
        fineX = 2*Xdim
        fineY = 2*Ydim
        # restrict only 2x2 blocks as in python side we make data dense.
        # Average 2x2 blocks in x,y for all z
        coarse_f[:] = (
            fine_r[0:fineX:2, 0:fineY:2, :] +
            fine_r[0:fineX:2, 1:fineY:2, :] +
            fine_r[1:fineX:2, 0:fineY:2, :] +
            fine_r[1:fineX:2, 1:fineY:2, :]
        ) / 4.0


    def interpolate(self, coarse_level: int):
        """Semicoarsening interpolation - only interpolate in 2D (x,y), keep z dimension"""
        coarse = self.grids[coarse_level]
        fine = self.grids[coarse_level - 1]
        
        coarse_u = coarse['u']
        fine_u = fine['u']
        
        # If same size, just copy
        if (fine['nx'] == coarse['nx'] and 
            fine['ny'] == coarse['ny'] and 
            fine['nz'] == coarse['nz']):
            fine_u[:] += coarse_u[:]
            return
        
        # Semicoarsening interpolation: copy coarse value to 4 fine points in 2D
        # Keep z dimension unchanged (no interpolation in z)
        for i in range(coarse['nx']):
            for j in range(coarse['ny']):
                for k in range(coarse['nz']):
                    fi, fj = 2*i, 2*j  # Only double x,y coordinates
                    fk = k  # z coordinate stays the same
                    coarse_val = coarse_u[i, j, k]
                    
                    # Add coarse value to 4 fine points in 2D (x,y plane)
                    for di in [0, 1]:
                        for dj in [0, 1]:
                            if (fi+di < fine['nx'] and fj+dj < fine['ny'] and fk < fine['nz']):
                                fine_u[fi+di, fj+dj, fk] += coarse_val

    # def interpolate(self, coarse_level: int):
    #     """Semicoarsening interpolation: expand in x,y by 2, keep z unchanged."""
    #     coarse = self.grids[coarse_level]     # (nz, Ny, Nx)
    #     fine   = self.grids[coarse_level - 1] # (nz, ny, nx)

    #     coarse_u = coarse['u']
    #     fine_u   = fine['u']

    #     # If sizes match, just add and return
    #     if (fine['nx'] == coarse['nx'] and fine['ny'] == coarse['ny'] and fine['nz'] == coarse['nz']):
    #         fine_u[:] += coarse_u[:]
    #         return

    #     Nz, Ny, Nx = coarse_u.shape

    #     # Make a 2x upsampled block in y and x, then add into the matching fine region
    #     up_yx = coarse_u.repeat(2, axis=1).repeat(2, axis=2)   # (nz, 2*Ny, 2*Nx)
    #     fine_u[:, :2*Ny, :2*Nx] += up_yx
    
    def solve_coarse(self, level: int):
        """Solve on coarsest level using Jacobi"""
        grid = self.grids[level]
        
        # Many Jacobi iterations on coarsest level
        for _ in range(50):
            self.jacobi_smooth(level, self.BOTTOM_SOLVER_ITER)
    
    def v_cycle(self, level: int = 0):
        """V-cycle multigrid"""
        if level == self.num_levels - 1:
            # Coarsest level: solve directly
            self.solve_coarse(level)
            return
        
        # Pre-smooth
        self.jacobi_smooth(level, self.PRE_SMOOTH_ITER)
        
        # Compute residual
        self.compute_residual(level)
        
        # Restrict residual to coarser level
        self.restrict(level)
        
        # Initialize coarse solution
        self.grids[level + 1]['u'].fill(0.0)
        
        # Recursive call to coarser level
        self.v_cycle(level + 1)
                
        # Interpolate correction
        self.interpolate(level + 1)
        
        # Post-smooth
        self.jacobi_smooth(level, self.POST_SMOOTH_ITER)
    
    def solve(self, max_iter: int = 20):
        """Solve using V-cycles"""
        # Initialize solution
        for grid in self.grids:
            grid['u'].fill(0.0)
        
        print(f"Starting GMG solve with {max_iter} max iterations")
        print(f"Grid size: {self.nx}x{self.ny}x{self.nz}, Levels: {self.num_levels}")
        print(f"Tolerance: {self.tolerance}")
        print("-" * 50)
        
        start_time = time.time()
        iterations = 0
        residual = 100.0
        
        while residual > self.tolerance and iterations < max_iter:
            # Perform V-cycle
            cycle_start = time.time()
            self.v_cycle()
            cycle_time = time.time() - cycle_start
            
            # Check convergence
            self.compute_residual(0)
            residual = np.max(np.abs(self.grids[0]['r']))
            iterations += 1
            
            print(f"Iteration {iterations:2d}: Residual = {residual:.6e}, Time = {cycle_time:.4f}s")
        
        total_time = time.time() - start_time
        
        print("-" * 50)
        print(f"After {iterations} iterations")
        print(f"Final residual: {residual:.6e}")
        print(f"Tolerance: {self.tolerance:.6e}")
        converged = "Yes" if residual < self.tolerance else "No"
        print(f"Converged: {converged}")
        print(f"Total time: {total_time:.4f}s")
        
        return residual, iterations


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Simple GMG Solver')
    parser.add_argument('-s', '--size', default='32,32,32', 
                       help='Grid size as nx,ny,nz (default: 32,32,32)')
    parser.add_argument('-l', '--levels', type=int, default=4,
                       help='Number of multigrid levels (default: 4)')
    parser.add_argument('-n', '--max_iter', type=int, default=20,
                       help='Maximum iterations (default: 20)')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Print detailed level information')
    parser.add_argument('--tolerance', type=float, default=1e-6,
                       help='Convergence tolerance (default: 1e-6)')
    parser.add_argument('--pre-iter', type=int, default=6,
                       help='Number of pre-smoothing iterations (default: 6)')
    parser.add_argument('--post-iter', type=int, default=6,
                       help='Number of post-smoothing iterations (default: 6)')
    parser.add_argument('--bottom-iter', type=int, default=100,
                       help='Number of bottom solver iterations (default: 100)')
    
    args = parser.parse_args()
    
    # Parse grid size
    size_parts = args.size.split(',')
    if len(size_parts) != 3:
        raise ValueError("Grid size must be in format nx,ny,nz")
    
    nx, ny, nz = map(int, size_parts)
    
    print("=" * 60)
    print("Simple Geometric Multigrid Solver")
    print("=" * 60)
    
    # Create and run solver
    solver = SimpleGMG(nx, ny, nz, args.levels, args.verbose, args.tolerance, args.pre_iter, args.post_iter, args.bottom_iter)
    residual, iterations = solver.solve(args.max_iter)
    
    print(f"Final residual: {residual:.6e}")
    print(f"Iterations: {iterations}")


if __name__ == "__main__":
    main()
