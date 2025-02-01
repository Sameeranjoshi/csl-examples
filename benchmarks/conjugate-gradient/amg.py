import numpy as np
from numpy import linalg as LA
from scipy import sparse as sp
import scipy.sparse.linalg as spla
import pyamg    # Maybe we need this, try if it's modular?
## Textbook implementation


# solve a linear system A * x = b
# where A is a symmetric positive definite matrix
# The AMG algorithm is from pyamg, https://github.com/pyamg/pyamg

def pyamg(A, x, b, relative_tol, max_ite, cycle_type='V', accel_type='cg'):
    import pyamg
    from scipy.sparse import random

    # Use PyAMG for solving the system
    ml = pyamg.ruge_stuben_solver(A)  # Multi-level solver

    # Solve the system
    # Use relative tol as this solver expects it, read documentation.
    x_pyamg, info = ml.solve(b, tol=relative_tol, cycle=cycle_type, maxiter=max_ite, accel=accel_type, return_info=True)
    if info != 0:
        print("PyAMG did not converge, maybe try changing it's parameters. halted at iterations: ", info)

    r = b - A.dot(x_pyamg)
    rho = np.dot(r, r)
    return x_pyamg, rho

def scipy_direct_solver(A, b):
    x = spla.spsolve(A, b)
    r = b - A.dot(x)
    rho = np.dot(r, r)
    return x, rho

def scipy_iterative_solver(A, x, b, atol, max_ite):
    x, converged = spla.cg(A, b, x0=x, atol=atol, maxiter=max_ite)
    if converged > 0:
        print("spla.cg from scipy_iterative_solver did not converge, maybe use more iterations or reduce tolerance.")
    
    # calculate residual
    r = b - A.dot(x)
    rho = np.dot(r, r)
    return x, rho

#########Visit later
def doAMG_V_cycle(A_csr, x0, b, max_ite, tol):
    from pyamg import smoothed_aggregation_solver, ruge_stuben_solver
    from scipy.sparse.linalg import cg
    
    print("AMG solver")
    print(f"tolerance: {tol}")  
    ml = ruge_stuben_solver(A_csr, max_coarse=20, max_levels=20, coarse_solver='gmres')
    residuals = []
    x = ml.solve(b=b, x0=x0, tol=tol, residuals=residuals, cycle='V', maxiter=max_ite)
    
    # compute the residual
    rho = [np.dot(r, r) for r in residuals]
    # print each rho
    for i, r in enumerate(rho):
        print(f"[AMG] levels {i}: rho = {r}")
    return x, rho[-1], max_ite    # last redisual

class AMGVCycle:
    def __init__(self, A, levels=3, iterations=3, tol=1.e-5):
        self.A = A
        self.levels = levels
        self.iterations = iterations
        self.tol = tol
    
    def jacobi_smooth(self, x, b, iterations=3):
        """Simple Jacobi smoother."""
        D = self.A.diagonal()
        for _ in range(iterations):
            x = (b - (self.A @ x - D*x)) / D
        return x

    def coarsen(self):
        """Simple C/F splitting based on the diagonal dominance."""
        n = self.A.shape[0]
        coarse_idx = np.arange(0, n, 2)  # Select every second row as coarse
        fine_idx = np.setdiff1d(np.arange(n), coarse_idx)
        return coarse_idx, fine_idx

    def restrict(self, fine_idx, coarse_idx):
        """Simple direct injection restriction."""
        n, nc = len(fine_idx), len(coarse_idx)
        R = sp.lil_matrix((nc, n + nc))
        for i, ci in enumerate(coarse_idx):
            R[i, ci] = 1  # Direct injection
        return R.tocsr()

    def interpolate(self, R):
        """Piecewise constant interpolation (transpose of restriction)."""
        return R.T

    def solve(self, x, b, levels=None):
        """AMG V-cycle solver."""
        if levels is None:
            levels = self.levels
        
        if levels == 0 or self.A.shape[0] < 2:
            x, info = spla.cg(self.A, b, x0=x, tol=self.tol, maxiter=self.iterations)
            return x
        
        # Pre-smoothing
        x = self.jacobi_smooth(x, b, iterations=self.iterations)
        
        # Coarsening
        coarse_idx, fine_idx = self.coarsen()
        R = self.restrict(fine_idx, coarse_idx)
        P = self.interpolate(R)
        A_coarse = R @ self.A @ P
        b_coarse = R @ (b - self.A @ x)
        
        # Recursive call to solve on coarse grid
        x_coarse = np.zeros(len(coarse_idx))
        x_coarse = AMGVCycle(A_coarse, levels - 1).solve(x_coarse, b_coarse)
        
        # Coarse-grid correction
        x += P @ x_coarse
        
        # Post-smoothing
        x = self.jacobi_smooth(x, b, iterations=self.iterations)
        
        return x

class TwoGridMultigrid:
    def __init__(self, A, p1=3, p2=3, omega=2/3):
        self.A = A
        self.p1 = p1  # Pre-smoothing steps
        self.p2 = p2  # Post-smoothing steps
        self.omega = omega
    
    def weighted_jacobi(self, x, b, iterations):
        """Weighted Jacobi smoothing."""
        D = self.A.diagonal()
        for _ in range(iterations):
            x += self.omega * (b - self.A @ x) / D
        return x
    
    def coarsen(self):
        """Simple coarsening strategy using direct injection."""
        n = self.A.shape[0]
        coarse_idx = np.arange(0, n, 2)
        fine_idx = np.setdiff1d(np.arange(n), coarse_idx)
        return coarse_idx, fine_idx
    
    def restrict(self, fine_idx, coarse_idx):
        """Restriction operator using direct injection."""
        n, nc = len(fine_idx), len(coarse_idx)
        R = sp.lil_matrix((nc, n + nc))
        for i, ci in enumerate(coarse_idx):
            R[i, ci] = 1
        return R.tocsr()
    
    def prolongate(self, R):
        """Prolongation operator is the transpose of restriction."""
        return R.T
    
    def solve(self, x, b):
        """Two-grid V-cycle multigrid solver."""
        # Pre-smoothing
        x = self.weighted_jacobi(x, b, self.p1)
        
        # Compute residual
        r = b - self.A @ x
        
        # Restriction to coarse grid
        coarse_idx, fine_idx = self.coarsen()
        R = self.restrict(fine_idx, coarse_idx)
        P = self.prolongate(R)
        A_coarse = R @ self.A @ P
        r_coarse = R @ r
        
        # Solve on coarse grid
        x_coarse = np.zeros(len(coarse_idx))
        x_coarse = spla.spsolve(A_coarse, r_coarse)
        
        # Prolongation to fine grid
        z = P @ x_coarse
        
        # Update solution
        x += z
        
        # Post-smoothing
        x = self.weighted_jacobi(x, b, self.p2)
        
        return x
