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
    # D_inv = np.diag(1 / np.diag(A))  # Inverse of diagonal matrix
    # Extract diagonal and create inverse
    D = np.diag(A)
    # Create a mask for non-zero diagonal elements
    non_zero_mask = D != 0
    # Create inverse with zeros for zero diagonal elements
    D_inv = np.zeros((len(D), len(D)), dtype=np.float32)
    D_inv[non_zero_mask, non_zero_mask] = 1.0 / D[non_zero_mask]
        
    print(f"A shape: {A.shape}")
    print(f"b shape: {b.shape}")
    print(f"x shape: {x.shape}")
    print(f"D_inv shape: {D_inv.shape}")

    for iter in range(iterations):
        term1 = omega * np.dot(D_inv, b)
        print(f"term1 shape: {term1.shape}")
        
        eye_matrix = np.eye(N)
        print(f"eye_matrix shape: {eye_matrix.shape}")
        
        d_inv_a = np.dot(D_inv, A)
        print(f"D_inv*A shape: {d_inv_a.shape}")
        
        term2_matrix = eye_matrix - omega * d_inv_a
        print(f"(I-omega*D_inv*A) shape: {term2_matrix.shape}")
        
        term2 = np.dot(term2_matrix, x)
        print(f"term2 shape: {term2.shape}")
        
        x = term1 + term2
        print(f"Iteration {iter+1}, x: {x}")
        print(f"x shape: {x.shape}")

    return x

# Example matrix A (4x4), vector b, and initial x
# A = np.array([[4, -1,  0,  0],
#               [-1, 4, -1, 0],
#               [0, -1, 4, -1],
#               [0, 0, -1, 4]], dtype=np.float32)
# b = np.array([1, 2, 0, 1], dtype=np.float32)
# x = np.ones(4, dtype=np.float32)

# A = np.array([[4, 1,  3,  3],
#               [4, 1, 0, 1],
#               [2, 2, 0, 4],
#               [0, 4, 0, 3]], dtype=np.float32)
# b = np.array([0,0,0,3], dtype=np.float32)
# x = np.array([2,3,4,4], dtype=np.float32)

# iterations = 1
# jacobi_omega = 1.0
# input_x = np.copy(x)
# input_x_alt = np.copy(x)
# print("Input x:", input_x)
# print("Input x alt:", input_x_alt)

# smoothed_x = jacobi_iteration(A, b, input_x, jacobi_omega, iterations)
# smoothed_x_alt = jacobi_iteration_alt(A, b, input_x_alt, jacobi_omega, iterations)

# print("Smoothed x:", smoothed_x)
# print("Smoothed x alt:", smoothed_x_alt)





########################################
########################################
import numpy as np


def pad_A(A, kernel_rows, kernel_cols):
    # Only pad if dimensions are not divisible by kernel size
    if A.shape[0] % kernel_rows == 0 and A.shape[1] % kernel_cols == 0:
        A_padded = A
    else:
        pad_rows = kernel_rows - A.shape[0] % kernel_rows if A.shape[0] % kernel_rows != 0 else 0
        pad_cols = kernel_cols - A.shape[1] % kernel_cols if A.shape[1] % kernel_cols != 0 else 0
        A_padded = np.pad(A, ((0, pad_rows), (0, pad_cols)), mode='constant', constant_values=0)
    return A_padded
def pad_1d(vector, padded_size):
    if vector.shape[0] == 1:
        # Add zeros to reach padded_size
        pad_size = padded_size - vector.shape[0]
        # Reshape vector to column vector before padding
        vector = vector.reshape(-1, 1)
        vector_padded = np.pad(vector, ((0, pad_size), (0, 0)), mode='constant', constant_values=0)
    else:
        # Add zeros to reach padded_size
        pad_size = padded_size - vector.shape[0]
        # Reshape vector to column vector before padding
        vector = vector.reshape(-1, 1)
        vector_padded = np.pad(vector, ((0, pad_size), (0, 0)), mode='constant', constant_values=0)
    return vector_padded

# layouts
# Row-Row Layout: Row-major blocks, Row-major inside blocks
def row_row_layout(A, kernel_rows, kernel_cols):
    A_vsplit = np.vsplit(A, kernel_rows)  # Split into kernel_rows blocks
    A_blocks = []

    for block in A_vsplit:
        A_block = np.hsplit(block, kernel_cols)
        A_block_row_major = [b.ravel() for b in A_block]
        A_blocks.extend(A_block_row_major)

    return np.concatenate(A_blocks)

# Row-Col Layout: Split A into blocks, store each block in column-major order
def row_col_layout(A, kernel_rows, kernel_cols):
    A_vsplit = np.vsplit(A, kernel_rows)  # Split into kernel_rows blocks
    A_blocks_col_major = []

    for block in A_vsplit:
        # Split each row block into column blocks
        A_block = np.hsplit(block, kernel_cols)
        # Convert each sub-block to column-major order by transposing and flattening
        A_block_col_major = [b.T.ravel() for b in A_block]
        A_blocks_col_major.extend(A_block_col_major)  # Add blocks one after another

    # Concatenate all blocks
    return np.concatenate(A_blocks_col_major)

# Col-Row Layout: Column-major blocks, Row-major inside blocks
def col_row_layout(A, kernel_rows, kernel_cols):
    A_hsplit = np.hsplit(A, kernel_cols)  # Split into kernel_cols blocks first
    A_blocks = []

    for block in A_hsplit:
        A_block = np.vsplit(block, kernel_rows)
        A_block_row_major = [b.ravel() for b in A_block]
        A_blocks.extend(A_block_row_major)

    return np.concatenate(A_blocks)

# Col-Col Layout: Column-major blocks, Column-major inside blocks
def col_col_layout(A, kernel_rows, kernel_cols):
    A_hsplit = np.hsplit(A, kernel_cols)  # Split into kernel_cols blocks first
    A_blocks = []

    for block in A_hsplit:
        A_block = np.vsplit(block, kernel_rows)
        A_block_col_major = [b.T.ravel() for b in A_block]
        A_blocks.extend(A_block_col_major)

    return np.concatenate(A_blocks)


########################################
# 1. input problem shape, input PE shape
# 2. Pad if necessary
# 3. Layouts
########################################

# # Sample matrix A
# import sys
# M = int(sys.argv[1])
# N = int(sys.argv[2])  # Matrix dimensions
# kernel_rows = int(sys.argv[3])
# kernel_cols = int(sys.argv[4])

# A = np.arange(M*N, dtype=np.float32).reshape(M,N).astype(np.float32)
# X = np.arange(N, dtype=np.float32).reshape(N,1).astype(np.float32)
# B = np.arange(M, dtype=np.float32).reshape(M,1).astype(np.float32)

# A_padded = pad_A(A, kernel_rows, kernel_cols)
# [padded_M, padded_N] = A_padded.shape
# X_padded = pad_1d(X, padded_N)
# B_padded = pad_1d(B, padded_M)

# # per PE - floor division operator (//) rounds down to nearest integer
# per_pe_M = padded_M // kernel_rows  # Floor division - rounds down padded matrix rows / kernel rows
# per_pe_N = padded_N // kernel_cols  # Floor division - rounds down padded matrix cols / kernel cols

# # Test all layouts
# print("kernel_rows:", kernel_rows, "kernel_cols:", kernel_cols)
# print("Original A shape:", A.shape, "--> Padded A shape:", A_padded.shape, "per PE shape:", (per_pe_M, per_pe_N))
# print("Original X shape:", X.shape, "--> Padded X shape:", X_padded.shape)
# print("Original B shape:", B.shape, "--> Padded B shape:", B_padded.shape)

# # layouts
# A_row_row = row_row_layout(A_padded, kernel_rows, kernel_cols)
# A_row_col = row_col_layout(A_padded, kernel_rows, kernel_cols)
# A_col_row = col_row_layout(A_padded, kernel_rows, kernel_cols)
# A_col_col = col_col_layout(A_padded, kernel_rows, kernel_cols)

# print("\nRow-Row Layout (Row-major blocks, Row-major inside blocks):")
# print(A_row_row)

# print("\nRow-Col Layout (Row-major blocks, Column-major inside blocks):")
# print(A_row_col)

# print("\nCol-Row Layout (Column-major blocks, Row-major inside blocks):")
# print(A_col_row)

# print("\nCol-Col Layout (Column-major blocks, Column-major inside blocks):")
# print(A_col_col)
