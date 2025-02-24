import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import pyamg
import matplotlib.pyplot as plt
import utilities as ut
import cg as CG




# Set random seed for reproducibility
np.random.seed(42)
N = 10
sparsity = 0.90  # 90% sparse


# A_csr_no_SPD = sp.random(N, N, density=(1 - sparsity), format="csr", dtype=np.float32)ix(A_csr_no_SPD, title="Random Matrix no SPD", filename="random_nospd.png")
# A_csr_SPD = make_matrix_SPD(A_csr_no_SPD)
# ut.visualize_matrix(A_csr_SPD, title="Random Matrix SPD", filename="random_spd.png")
 
from sklearn.datasets import make_sparse_spd_matrix
A_SPD = make_sparse_spd_matrix(n_dim=N,
                                    norm_diag=True,
                                    smallest_coef=0.1,
                                    largest_coef=0.9,
                                    random_state=0)
A_csr_SPD = sp.csr_matrix(A_SPD)
ut.visualize_matrix(A_csr_SPD, title="A Matrix SPD", filename="A_spd.png")



# using a poisson matrix, to be compatible with pyamg
A_csr = A_csr_SPD   
b = np.random.rand(A_csr.shape[0]).astype(np.float32)   # random right-hand side vector
x0 = np.zeros(A_csr.shape[1], dtype=np.float32)    # initial guess
# print shapes of all
print("Matrix Shape:", A_csr.shape)
print("RHS Shape:", b.shape)
print("Initial Guess Shape:", x0.shape)

################################################################################
# INTERFACE A_csr, b, x0
################################################################################
# pyamg or something else.

ml = ut.setup(A_csr, b)
print(ml)                                           # print hierarchy information

A_coarse = ml.levels[-1].A
A_fine = ml.levels[0].A
ut.visualize_matrix(A_fine, title="Fine Matrix", filename="fine.png")
ut.visualize_matrix(A_coarse, title="Coarse Matrix", filename="coarse.png")
print("[pyamg]Is coarse SPD:", ut.isspd(A_coarse))
print("[pyamg]Is fine SPD:", ut.isspd(A_fine))

ml.solve(b, x0, maxiter=100, tol=1e-10, accel="cg")
print(f"[pyamg-cg]Residual:", np.linalg.norm(b - A_csr.dot(x0)))
ml.solve(b, x0, maxiter=100, tol=1e-10, accel="bicgstab")
print(f"[pyamg-bicgstab]Residual:", np.linalg.norm(b - A_csr.dot(x0)))


xcg, rhocg, kcg = CG.conjugateGradient(A_csr, x0, b, 100, 1e-10)
print(f"[CG]Residual:", rhocg)
print(f"[CG]Number of iterations:", kcg)
print(f"[CG]Residual:", np.linalg.norm(b - A_csr.dot(xcg)))

xc, info00 = spla.cg(A_csr, b, x0, rtol=1e-10)
# converged?
print(f"[scipy-cg]Converged:", info00)
print(f"[scipy-cg]Residual:", np.linalg.norm(b - A_csr.dot(xc)))

xv, info01 = spla.bicgstab(A_csr, b, x0=x0, rtol=1e-10)
# converged?
print(f"[scipy-bicgstab]Converged:", info01)
print(f"[scipy-bicgstab]Residual:", np.linalg.norm(b - A_csr.dot(xv)))

