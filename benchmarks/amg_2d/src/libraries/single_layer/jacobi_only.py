import numpy as np

# Jacobi iteration logic
# formula: x = (1-omega)*x + omega*(b - A*x)/diag(A)
# Ax is for all values except the diagonal value.
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

# Jacobi iteration logic (new formula)
# formula: x = omega*D_inv*b + (I-omega*D_inv*A)*x
def jacobi_iteration_alt(A, b, x, omega, iterations):
    N = len(b)
    D_inv = np.diag(1 / np.diag(A))  # Inverse of diagonal matrix

    for iter in range(iterations):
        term1 = omega * np.dot(D_inv, b)
        term2 = np.dot((np.eye(N) - omega * np.dot(D_inv, A)), x)
        x = term1 + term2
        print(f"Iteration {iter+1}, x: {x}")

    return x

# Example matrix A (4x4), vector b, and initial x
A = np.array([[4, -1,  0,  0],
              [-1, 4, -1, 0],
              [0, -1, 4, -1],
              [0, 0, -1, 4]], dtype=np.float32)
b = np.array([1, 2, 0, 1], dtype=np.float32)
x = np.ones(4, dtype=np.float32)

iterations = 1
jacobi_omega = 1.0
input_x = np.copy(x)
input_x_alt = np.copy(x)
print("Input x:", input_x)
print("Input x alt:", input_x_alt)

smoothed_x = jacobi_iteration(A, b, input_x, jacobi_omega, iterations)
smoothed_x_alt = jacobi_iteration_alt(A, b, input_x_alt, jacobi_omega, iterations)

print("Smoothed x:", smoothed_x)
print("Smoothed x alt:", smoothed_x_alt)
