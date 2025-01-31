import numpy as np
from numpy import linalg as LA
from scipy import sparse as sp
import pyamg

# solve a linear system A * x = b
# where A is a symmetric positive definite matrix
# The AMG algorithm is from pyamg, https://github.com/pyamg/pyamg

def scipy_CG(A_csr, x0, b, max_ite, tol):

    # This line below makes both implementation related differences in `tol` equal.
    # From both CG and scipy implementations.
    # Scipy = L2 norm, CG = L2 squared norm.
    # Not sure why is this divided then, shouldn't it be multiplied?
    tol = tol / np.linalg.norm(b, 2)
    print(f"tolerance(tol/||b||): {tol}")

    from scipy.sparse.linalg import cg, cgs, gmres, bicg, bicgstab
    iteration_count = 0  # Track iterations

    r = b - A_csr.dot(x0)
    rho = np.dot(r, r)
    print(f"[SCIPY(CG)] iter {iteration_count}: rho = {rho}")

    # Callback function for iteration count
    def count_iterations(xk):
        nonlocal iteration_count
        iteration_count += 1  # Correctly track iterations
        r = b - A_csr.dot(xk)
        rho = np.dot(r, r)
        print(f"[SCIPY(CG)] iter {iteration_count}: rho = {rho}")

    # Call SciPy CG solver
    x, info = sp.linalg.cg(A_csr, b, x0=x0, tol=tol, maxiter=max_ite, callback=count_iterations)
    
    # find rho
    r = b - A_csr.dot(x)
    rho = np.dot(r, r)
    return x, rho, iteration_count

def doAMG_V_cycle(A_csr, x0, b, max_ite, tol):
    
    from pyamg import smoothed_aggregation_solver, ruge_stuben_solver
    from scipy.sparse.linalg import cg
    
    print("AMG solver")
    ml = ruge_stuben_solver(A_csr, max_coarse=20, max_levels=20, coarse_solver='gmres')
    residuals = []
    x = ml.solve(b=b, x0=x0, tol=tol*tol, residuals=residuals, cycle='V', maxiter=max_ite)
    
    # compute the residual
    rho = [np.dot(r, r) for r in residuals]
    # print each rho
    for i, r in enumerate(rho):
        print(f"[AMG] iter {i}: rho = {r}")
    return x, rho[-1], max_ite    # last redisual
