import numpy as np
import amg as ag


# Example usage
A = np.array([[4, -1, 0], [-1, 4, -1], [0, -1, 4]], dtype=float)
b = np.array([2, 6, 2], dtype=float)
x0 = np.zeros_like(b)

smoother_obj = ag.smoother(A, b)
pyamg_jacobi = smoother_obj.pyamg_jacobi(x0, max_iter=100, weight=1.0)
pyamg_gauss_seidel = smoother_obj.pyamg_gauss_seidel(x0, max_iter=100)
pyamg_sor = smoother_obj.pyamg_sor(x0, max_iter=100, weight=1.0)
simple_jacobi = smoother_obj.simple_jacobi(x0, max_iter=100)
gauss_seidel = smoother_obj.gauss_seidel(x0, max_iter=100)

print(pyamg_jacobi) 
print(pyamg_gauss_seidel)
print(pyamg_sor)
print(simple_jacobi)
print(gauss_seidel)
# Output: [0.5 1.5 0.5]