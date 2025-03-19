import numpy as np

# Jacobi iteration logic
def jacobi_iteration(A, b, x, omega, iterations):
    N = len(b)
    for iter in range(iterations):
        tmp = x.copy()
        for i in range(N):
            rsum = 0
            for j in range(N):
                if i != j:
                    rsum += A[i, j] * tmp[j]

            if A[i, i] != 0:
                x[i] = (1 - omega) * tmp[i] + omega * (b[i] - rsum) / A[i, i]
        
        print(f"Iteration {iter+1}, x: {x}")
    
    return x

# # Example matrix A (4x4), vector b, and initial x
# A = np.array([[4, -1,  0,  0],
#               [-1, 4, -1, 0],
#               [0, -1, 4, -1],
#               [0, 0, -1, 4]], dtype=np.float32)
# b = np.array([1, 2, 0, 1], dtype=np.float32)
# x = np.ones(4, dtype=np.float32)

# iterations = 1
# jacobi_omega = 1.0
# input_x = np.copy(x)
# print("Input x:", input_x)
# smoothed_x = jacobi_iteration(A, b, x, jacobi_omega, iterations)

