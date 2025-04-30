import numpy as np

def jacobi_csr(A_Csr, x, b, omega=1.0/3.0, iterations=1):
    """
    Perform one iteration of the Jacobi method for solving Ax = b using CSR matrix.

    Parameters:
    A_Csr : csr_matrix
        Sparse matrix in CSR format representing the system of equations.
    x : ndarray
        Approximate solution vector.
    b : ndarray
        Right-hand side vector.
    omega : float
        Damping parameter.
    iterations : int
        Number of Jacobi iterations to perform.

    Returns:
    x : ndarray
        Updated solution vector.
    """
    # if not sp.isspmatrix_csr(A_Csr):
    #     A_Csr = A_Csr.tocsr()
        
    for _ in range(iterations):
        # Create a temporary vector to hold the previous values of x
        temp = np.copy(x)
        # Iterate over all the rows of the CSR matrix
        for i in range(A_Csr.shape[0]):  # Iterate over rows
            rsum = 0.0
            diag = 0.0

            # Get the start and end indices of the non-zero elements for row i
            start = A_Csr.indptr[i]
            end = A_Csr.indptr[i + 1]

            # Iterate over the non-zero elements in row i
            for jj in range(start, end):
                j = A_Csr.indices[jj]  # Column index of non-zero element
                value = A_Csr.data[jj]  # Value of the non-zero element

                if i == j:
                    diag = value  # Diagonal element for this row
                else:
                    rsum += value * temp[j]  # Off-diagonal terms

            # Update x[i] based on the Jacobi formula
            if diag != 0.0:
                x[i] = (1 - omega) * temp[i] + omega * (b[i] - rsum) / diag
    return x
