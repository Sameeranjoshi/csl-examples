#!/usr/bin/env python3
"""
Simplest Geometric Multigrid Solver for 3D Poisson equation (stable)
- Dirichlet boundary conditions (u = 0 on boundary)
- Semicoarsening in x,y (z unchanged)
- Weighted Jacobi smoother (ω = 2/3 default)
"""

import numpy as np
import time
from typing import Tuple, List

DTYPE = np.float32

class SimpleGMG:
    """Simplest stable GMG solver for 3D Poisson equation"""

    def __init__(self, nx: int, ny: int, nz: int, num_levels: int = 4,
                 verbose: bool = False, tolerance: float = 1e-6,
                 pre_iter: int = 4, post_iter: int = 3, bottom_iter: int = 150):
        """
        Args:
            nx, ny, nz: Grid dimensions
            num_levels: Number of multigrid levels
            verbose: Whether to print detailed level information
            tolerance: Convergence tolerance (compared against ||r||^2)
            pre_iter: Number of pre-smoothing iterations
            post_iter: Number of post-smoothing iterations
            bottom_iter: Number of bottom solver iterations (Jacobi)
        """
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.num_levels = num_levels
        self.verbose = verbose

        # Weighted Jacobi parameter (typical for 3D Laplacian)
        self.omega = DTYPE(2.0 / 3.0)

        self.tolerance = tolerance

        # Stencil coefficients for ∇^2 (discrete Laplacian)
        self.ALPHA = DTYPE(-6.0)  # center
        self.BETA  = DTYPE( 1.0)  # neighbors

        # Iteration counts
        self.PRE_SMOOTH_ITER   = pre_iter
        self.POST_SMOOTH_ITER  = post_iter
        self.BOTTOM_SOLVER_ITER = bottom_iter

        # Create grid hierarchy and initialize RHS
        self.grids = self._create_grids()
        self._init_rhs()

        if self.verbose:
            self._print_level_info()
            self._print_initialization_info()

    # ----------------------------
    # Setup / Introspection helpers
    # ----------------------------
    def _create_grids(self):
        """Create grid hierarchy with semicoarsening (reduce x,y; keep z)."""
        grids = []
        for level in range(self.num_levels):
            nx = max(2, self.nx // (2 ** level))
            ny = max(2, self.ny // (2 ** level))
            nz = self.nz  # keep z unchanged

            # uniform spacing in [0,1]; use interior spacing
            h = DTYPE(1.0) / DTYPE(max(nx - 1, 1))

            grid = {
                'nx': nx, 'ny': ny, 'nz': nz, 'h': h,
                'u':   np.zeros((nx, ny, nz), dtype=DTYPE),  # solution
                'f':   np.zeros((nx, ny, nz), dtype=DTYPE),  # RHS
                'r':   np.zeros((nx, ny, nz), dtype=DTYPE),  # residual
                'Au':  np.zeros((nx, ny, nz), dtype=DTYPE),  # A*u
                'rho': DTYPE(0.0),
                'rho_up': DTYPE(0.0),
            }
            grids.append(grid)
        return grids

    def _print_level_info(self):
        print("\nMultigrid Level Hierarchy:")
        print("=" * 50)
        for level, g in enumerate(self.grids):
            print(f"Level {level:2d}: {g['nx']:3d} x {g['ny']:3d} x {g['nz']:3d} "
                  f"(h = {g['h']:.6f}, points = {g['nx']*g['ny']*g['nz']:,})")
        print("=" * 50)

        print("\nData Structures at Each Level:")
        print("=" * 120)
        print(f"{'Level':<6} {'Grid Size (nx×ny×nz)':<22} {'u shape':<22} "
              f"{'f shape':<15} {'r shape':<15} {'Au shape':<15} {'Stencil':<20}")
        print("-" * 120)
        for level, g in enumerate(self.grids):
            print(f"{level:<6} "
                  f"{(g['nx'],g['ny'],g['nz'])!s:<22} "
                  f"{g['u'].shape!s:<22} {g['f'].shape!s:<15} "
                  f"{g['r'].shape!s:<15} {g['Au'].shape!s:<15} "
                  f"{'7-point Poisson':<20}")
        print("-" * 120)

        print("\nStencil Coefficients:")
        print("-" * 40)
        print(f"  α (center)   = {self.ALPHA}")
        print(f"  β (neighbor) = {self.BETA}")
        print("  Au[i,j,k] = α*u[i,j,k] + β*(±x, ±y, ±z neighbors)")
        print("-" * 40)

    def _print_initialization_info(self):
        print("\nInitialization Parameters:")
        print("=" * 50)
        print(f"Grid: {self.nx}×{self.ny}×{self.nz}, Levels: {self.num_levels}")
        print(f"Top-level h: {self.grids[0]['h']:.6f}")
        print("Boundary: Dirichlet (u=0 on domain boundary)")
        print("Smoother: Weighted Jacobi, ω=2/3")
        print("=" * 50)

    def _init_rhs(self):
        """Initialize right-hand side with f = sin(πx) sin(πy) sin(πz)."""
        for g in self.grids:
            nx, ny, nz = g['nx'], g['ny'], g['nz']
            x = np.linspace(0, 1, nx, dtype=DTYPE)
            y = np.linspace(0, 1, ny, dtype=DTYPE)
            z = np.linspace(0, 1, nz, dtype=DTYPE)
            X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
            pi = DTYPE(np.pi)
            g['f'] = np.sin(pi * X) * np.sin(pi * Y) * np.sin(pi * Z)

            # Zero RHS on boundary for pure Dirichlet forcing
            g['f'][0,:,:]  = 0; g['f'][-1,:,:] = 0
            g['f'][:,0,:]  = 0; g['f'][:,-1,:] = 0
            g['f'][:,:,0]  = 0; g['f'][:,:,-1] = 0

    # ----------------------------
    # Core linear algebra kernels
    # ----------------------------
    def calculate_rho(self, residual_3d: np.ndarray) -> float:
        """Compute ||r||^2 (squared L2 norm)."""
        r = residual_3d.ravel()
        return float(np.dot(r, r))

    def apply_operator(self, level: int):
        """
        Apply 7-point Laplacian: Au = (α*u + β*sum(neighbors)) / h^2.
        We compute only on interior points; boundary Au is kept at 0.
        """
        g = self.grids[level]
        u, Au, h = g['u'], g['Au'], g['h']
        nx, ny, nz = g['nx'], g['ny'], g['nz']
        alpha, beta = self.ALPHA, self.BETA

        Au.fill(0.0)

        # interior only
        for i in range(1, nx-1):
            for j in range(1, ny-1):
                ui_jm = u[i, j-1]
                ui_jp = u[i, j+1]
                for k in range(1, nz-1):
                    val = alpha * u[i, j, k]
                    val += beta * (u[i-1, j, k] + u[i+1, j, k] +
                                   ui_jm[k]     + ui_jp[k]     +
                                   u[i, j, k-1] + u[i, j, k+1])
                    Au[i, j, k] = val / (h*h)

        # boundaries remain zero (consistent with Dirichlet u=0)

    def compute_residual(self, level: int):
        """Compute residual r = f - Au; zeroed on boundary."""
        g = self.grids[level]
        self.apply_operator(level)
        g['r'][:] = g['f'] - g['Au']

        # boundary residuals = 0 (Dirichlet)
        g['r'][0,:,:]  = 0; g['r'][-1,:,:] = 0
        g['r'][:,0,:]  = 0; g['r'][:,-1,:] = 0
        g['r'][:,:,0]  = 0; g['r'][:,:,-1] = 0

    def _pin_boundary_zero(self, u: np.ndarray):
        u[0,:,:]  = 0; u[-1,:,:] = 0
        u[:,0,:]  = 0; u[:,-1,:] = 0
        u[:,:,0]  = 0; u[:,:,-1] = 0

    def jacobi_smooth(self, level: int, num_iter: int = 1):
        """Weighted Jacobi on interior; Dirichlet boundary pinned to 0."""
        g = self.grids[level]
        u, f, h = g['u'], g['f'], g['h']
        nx, ny, nz = g['nx'], g['ny'], g['nz']
        omega = self.omega

        diag = DTYPE(-6.0) / (h*h)  # interior diagonal of Laplacian

        for _ in range(num_iter):
            # compute r = f - Au
            self.apply_operator(level)
            # update = ω * (f - Au) / diag on interior
            update = omega * (f - g['Au']) / diag
            u[1:-1, 1:-1, 1:-1] += update[1:-1, 1:-1, 1:-1]
            # re-pin boundary
            self._pin_boundary_zero(u)

    # ----------------------------
    # Intergrid transfer operators
    # ----------------------------
    def restrict(self, fine_level: int):
        """Semicoarsening restriction (average 2x2 in x,y; z unchanged)."""
        fine = self.grids[fine_level]
        coarse = self.grids[fine_level + 1]

        fr = fine['r']
        cf = coarse['f']

        if (fine['nx'],fine['ny'],fine['nz']) == (coarse['nx'],coarse['ny'],coarse['nz']):
            cf[:] = fr[:]
            return

        Xc, Yc, Zc = cf.shape
        # fine extents we actually map from
        fineX = min(2 * Xc, fr.shape[0])
        fineY = min(2 * Yc, fr.shape[1])

        # average 2x2 block in x,y for all z
        cf[:, :, :] = (
            fr[0:fineX:2, 0:fineY:2, :] +
            fr[1:fineX:2, 0:fineY:2, :] +
            fr[0:fineX:2, 1:fineY:2, :] +
            fr[1:fineX:2, 1:fineY:2, :]
        ) / DTYPE(4.0)

        # enforce boundary on coarse RHS
        cf[0,:,:]  = 0; cf[-1,:,:] = 0
        cf[:,0,:]  = 0; cf[:,-1,:] = 0
        cf[:,:,0]  = 0; cf[:,:,-1] = 0

    def interpolate(self, fine_level: int):
        """Semicoarsening prolongation (inject coarse value to 2x2 in x,y; z unchanged)."""
        fine = self.grids[fine_level]
        coarse = self.grids[fine_level + 1]

        fu = fine['u']
        cu = coarse['u']

        if (fine['nx'],fine['ny'],fine['nz']) == (coarse['nx'],coarse['ny'],coarse['nz']):
            fu[:] += cu[:]
            self._pin_boundary_zero(fu)
            return

        Xc, Yc, Zc = cu.shape
        for i in range(Xc):
            fi = 2 * i
            if fi + 1 >= fu.shape[0]:
                continue
            for j in range(Yc):
                fj = 2 * j
                if fj + 1 >= fu.shape[1]:
                    continue
                # vectorize across z
                block = cu[i, j, :]
                fu[fi,   fj,   :] += block
                fu[fi+1, fj,   :] += block
                fu[fi,   fj+1, :] += block
                fu[fi+1, fj+1, :] += block

        # keep Dirichlet boundary after correction
        self._pin_boundary_zero(fu)

    # ----------------------------
    # Multigrid cycles
    # ----------------------------
    def v_cycle(self, level: int = 0):
        """Standard V-cycle."""
        if level == self.num_levels - 1:
            self.solve_coarse()
            return

        # pre-smooth
        self.jacobi_smooth(level, self.PRE_SMOOTH_ITER)

        # residual and restriction
        self.compute_residual(level)
        self.restrict(level)

        # coarse solve
        self.grids[level + 1]['u'].fill(0.0)
        self.v_cycle(level + 1)

        # prolongation and post-smooth
        self.interpolate(level)
        self.jacobi_smooth(level, self.POST_SMOOTH_ITER)

    def only_down_cycle(self, level: int = 0):
        """Downward sweep only (for split-cycle experiments)."""
        if level == self.num_levels - 1:
            return
        self.jacobi_smooth(level, self.PRE_SMOOTH_ITER)
        self.compute_residual(level)
        self.grids[level]['rho'] = self.calculate_rho(self.grids[level]['r'])
        self.restrict(level)
        self.grids[level + 1]['u'].fill(0.0)
        self.only_down_cycle(level + 1)

    def only_up_cycle(self, level: int):
        """Upward sweep only (for split-cycle experiments)."""
        if level == -1:
            return
        self.interpolate(level)
        self.jacobi_smooth(level, self.POST_SMOOTH_ITER)
        self.compute_residual(level)
        self.grids[level]['rho_up'] = self.calculate_rho(self.grids[level]['r'])
        self.only_up_cycle(level - 1)

    def solve_coarse(self):
        """Bottom solve (Jacobi) with Dirichlet pinning."""
        lvl = self.num_levels - 1
        for _ in range(self.BOTTOM_SOLVER_ITER):
            self.jacobi_smooth(lvl, 1)
        self.compute_residual(lvl)
        self.grids[lvl]['rho'] = self.calculate_rho(self.grids[lvl]['r'])

    # ----------------------------
    # User-facing solvers
    # ----------------------------
    def solve(self, max_iter: int = 20):
        """Solve using V-cycles; prints residual ||r||^2 each outer iter."""
        for g in self.grids:
            g['u'].fill(0.0)
            self._pin_boundary_zero(g['u'])

        print(f"Starting GMG solve with {max_iter} max iterations")
        print(f"Grid size: {self.nx}x{self.ny}x{self.nz}, Levels: {self.num_levels}")
        print(f"Tolerance (||r||^2): {self.tolerance}")
        print("-" * 50)

        start_time = time.time()
        iterations = 0
        residual = float('inf')

        for it in range(max_iter):
            t0 = time.time()
            self.v_cycle()
            cycle_time = time.time() - t0

            self.compute_residual(0)
            residual = self.calculate_rho(self.grids[0]['r'])
            iterations += 1

            print(f"Iteration {iterations:2d}: Residual = {residual:.6e}, Time = {cycle_time:.4f}s")
            if residual < self.tolerance:
                break

        total_time = time.time() - start_time
        print("-" * 50)
        print(f"After {iterations} iterations")
        print(f"Final residual (||r||^2): {residual:.6e}")
        print(f"Tolerance: {self.tolerance:.6e}")
        print(f"Converged: {'Yes' if residual < self.tolerance else 'No'}")
        print(f"Total time: {total_time:.4f}s")
        return residual, iterations

    def solve_iterative(self, max_iter: int = 20):
        """Split down/up cycle variant (optional)."""
        for g in self.grids:
            g['u'].fill(0.0)
            self._pin_boundary_zero(g['u'])

        print(f"Starting GMG solve with {max_iter} max iterations")
        print(f"Grid size: {self.nx}x{self.ny}x{self.nz}, Levels: {self.num_levels}")
        print(f"Tolerance (||r||^2): {self.tolerance}")
        print("-" * 50)

        start_time = time.time()
        iterations = 0
        residual = float('inf')

        for it in range(max_iter):
            t0 = time.time()
            self.only_down_cycle()
            self.solve_coarse()
            # propagate up
            self.grids[self.num_levels - 1]['rho_up'] = self.grids[self.num_levels - 1]['rho']
            self.only_up_cycle(self.num_levels - 2)
            cycle_time = time.time() - t0

            self.compute_residual(0)
            residual = self.calculate_rho(self.grids[0]['r'])
            iterations += 1

            print(f"Iteration {iterations:2d}: Residual = {residual:.6e}, Time = {cycle_time:.4f}s")
            if residual < self.tolerance:
                break

        total_time = time.time() - start_time
        print("-" * 50)
        print(f"After {iterations} iterations")
        print(f"Final residual (||r||^2): {residual:.6e}")
        print(f"Tolerance: {self.tolerance:.6e}")
        print(f"Converged: {'Yes' if residual < self.tolerance else 'No'}")
        print(f"Total time: {total_time:.4f}s")
        return residual, iterations

# Optional quick test
if __name__ == "__main__":
    # small smoke test
    solver = SimpleGMG(16, 16, 16, num_levels=3, verbose=False, tolerance=1e-3,
                       pre_iter=4, post_iter=3, bottom_iter=150)
    solver.solve(10)

