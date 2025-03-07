import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import pyamg
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import numpy as np

def generate_sparse_triangular_SPD(N, sparsity=0.90, diag_shift=10.0):
    """
    Generates a sparse SPD matrix using a sparse lower-triangular matrix.
    """
    # Generate a lower-triangular sparse matrix
    L = sp.random(N, N, density=(1 - sparsity), format="csr", dtype=np.float32)
    L = sp.tril(L)  # Keep only lower triangle

    # Make it symmetric: A = L + L.T
    A = L + L.T

    # Add diagonal shift for positive definiteness
    A.setdiag(A.diagonal() + diag_shift)

    return A

def generate_sparse_triangular_SPD(N, sparsity=0.90, diag_shift=10.0, format='csr'):
    """
    Generates a sparse SPD matrix using a sparse lower-triangular matrix
    and ensures it is in CSR format.

    Parameters:
    N (int): Matrix size (NxN)
    sparsity (float): Fraction of elements that should be zero
    diag_shift (float): Value to add to diagonal to ensure SPD
    format (str): Desired sparse matrix format (default: 'csr')

    Returns:
    scipy.sparse matrix: SPD matrix in the specified format.
    """
    # Generate a lower-triangular sparse matrix
    L = sp.random(N, N, density=(1 - sparsity), format="coo", dtype=np.float32)
    L = sp.tril(L)  # Keep only lower triangle

    # Make it symmetric: A = L + L.T
    A = L + L.T

    # Add diagonal shift for positive definiteness
    A.setdiag(A.diagonal() + diag_shift)

    # Convert to desired sparse format (default: 'csr')
    A = A.asformat(format)

    return A

def make_matrix_SPD(A_csr, add_diagonal=10.0):
    """
    Ensures a given sparse matrix is Symmetric Positive Definite (SPD) by:
    1. Making it symmetric: A = (A + A^T) / 2
    2. Adding a scaled identity matrix to ensure positive definiteness.

    Parameters:
    A (scipy.sparse matrix): Input sparse matrix.
    add_diagonal (float): Value to add to the diagonal for positive definiteness.

    Returns:
    scipy.sparse.csr_matrix: SPD matrix.
    """
    # Ensure symmetry
    A = A_csr.toarray()
    A_symmetric = (A + A.T) / 2

    # Add a multiple of the identity matrix to make it strictly positive definite
    A_spd = A_symmetric + add_diagonal * sp.eye(A.shape[0], format="csr", dtype=A.dtype)

    return sp.csr_matrix(A_spd)

def make_CSR(spd_matrix):
    spd_csr = sp.csr_matrix(spd_matrix)
    return spd_csr

def isspd(A_csr):
    A = A_csr.toarray()
    # check 1 - symmetric another method.
    is_symmetric = np.allclose(A, A.T)
    assert is_symmetric, "A must be symmetric"
    
    # check 2 - symmetric
    # print(A_csr)
    A_csc = A_csr.tocsc(copy=True)
    # A_csc.sort_indices()
    # A_csc = A_csc.astype(np.float32)
    assert 0 == np.linalg.norm(A_csr.indptr - A_csc.indptr, np.inf), "A must be symmetric"
    assert 0 == np.linalg.norm(A_csr.indices - A_csc.indices, np.inf), "A must be symmetric"
    assert 0 == np.linalg.norm(A_csr.data - A_csc.data, np.inf), "A must be symmetric"
        

    # check 3 - positive definite
    eigvals = np.linalg.eigvals(A)
    is_positive_definite = np.all(eigvals > 0)
    assert is_positive_definite, "A must be positive definite"

    print("Matrix is positive definite:", is_positive_definite)





def setup(A_csr, b=None):
    smoothed_aggregation_solver_config = {
        'max_levels': 2,
    }
    ml = pyamg.ruge_stuben_solver(A_csr, **smoothed_aggregation_solver_config)
    return ml


def visualize_matrix(A, title="Matrix", filename=None):
    """
    Visualizes a sparse matrix using a heatmap, only displaying nonzero values.

    Parameters:
    A (numpy array or scipy sparse matrix): The matrix to visualize.
    """
    if sp.issparse(A):
        A = A.toarray()

    plt.figure(figsize=(8, 6))

    # Create a mask to only show nonzero values
    mask = A != 0

    # Use masked array to hide zero values
    plt.imshow(np.where(mask, A, np.nan), cmap="viridis", aspect="auto", interpolation="nearest")

    plt.title(title)
    plt.xlabel("Columns")
    plt.ylabel("Rows")
    
    # Annotate only nonzero values for small matrices
    if A.shape[0] <= 10 and A.shape[1] <= 10:
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                if A[i, j] != 0:
                    plt.text(j, i, f"{A[i, j]:.2f}", ha="center", va="center", color="white")

    plt.colorbar(label="Value")
    plt.tight_layout()

    if filename:
        plt.savefig(filename)
    
    plt.show()

def visualize_matrix_or_vector(data, title="Data", filename=None):
    """
    Visualizes a sparse matrix or vector using a heatmap or line plot.

    Parameters:
    data (numpy array or scipy sparse matrix): The data to visualize.
    """
    if sp.issparse(data):
        data = data.toarray()

    if data.ndim == 1:
        plt.figure(figsize=(8, 6))
        data = data.reshape((int(np.sqrt(data.size)), -1))  # Reshape to square grid
        plt.imshow(data, cmap="viridis", aspect="auto", interpolation="nearest")
        plt.title(title)
        plt.xlabel("Columns")
        plt.ylabel("Rows")
        plt.colorbar(label="Value")
        if filename:
            plt.savefig(filename)
        plt.show()
    elif data.ndim == 2:
        visualize_matrix(data, title, filename)
    else:
        raise ValueError("Data must be either a 1D vector or 2D matrix.")



def hw_2_oned_colmajor(
    height: int,
    width: int,
    A_hw: np.ndarray,
    dtype
):
  """
    Given a 2-D tensor A[height][width], transform it to
    1D array by column-major
  """
  A_1d = np.zeros(height*width, dtype)
  idx = 0
  for w in range(width):
    for h in range(height):
      A_1d[idx] = A_hw[(h, w)]
      idx = idx + 1
  return A_1d


def transform_to_cliff_distribution(A, H, W):
    """
    Transforms a 2D tensor into a cliff distribution with column-major local tensor.

    Parameters:
    A (numpy.ndarray): Input 2D array of shape (H*W, H*W)
    H (int): Number of vertical partitions (rows of local blocks)
    W (int): Number of horizontal partitions (columns of local blocks)

    Returns:
    numpy.ndarray: Transformed array with column-major local tensor
    """
    # Step 1: Reshape into (H, W, local_H, local_W)
    local_H, local_W = A.shape[0] // H, A.shape[1] // W
    A1 = A.reshape(H, local_H, W, local_W)

    # Step 2: Transpose to (H, W, local_W, local_H) for column-major order
    A2 = A1.transpose(0, 2, 3, 1)

    # Step 3: Reshape back to 2D as (H*W, H*W)
    A3 = A2.reshape(H * W, local_H * local_W)

    return A3


def jacobi_dense(A, x, b, omega=1.0/3.0, iterations=1):
    """
    Perform one iteration of the Jacobi method for solving Ax = b using dense matrix.

    Parameters:
    A : ndarray
        Dense matrix representing the system of equations.
    x : ndarray
        Approximate solution vector.
    b : ndarray
        Right-hand side vector.
    omega : float
        Damping parameter.

    Returns:
    x : ndarray
        Updated solution vector.
    """
    for _ in range(iterations):
        # Create a temporary vector to hold the previous values of x
        temp = np.copy(x)

        # Iterate over all the rows of the matrix A
        for i in range(A.shape[0]):  # Iterate over rows (row_start = 0, row_stop = number of rows)
            rsum = 0.0
            diag = A[i, i]  # Diagonal element of A for row i

            # Compute the sum of off-diagonal terms
            for j in range(A.shape[1]):  # Iterate over columns
                if i != j:
                    rsum += A[i, j] * temp[j]

            # Update x[i] based on the Jacobi formula
            if diag != 0.0:
                x[i] = (1 - omega) * temp[i] + omega * (b[i] - rsum) / diag
    
    return x

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

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def visualize_layout_with_empty(fabric_dimensions, layer_coordinates_map, layer_params_map, total_pe_cols, total_pe_rows, filename="fabric_layout_with_empty.png"):
    """
    Visualizes the layout of layers on a fabric grid and marks empty spaces inside @set_rectangle(total_pe_cols, total_pe_rows).
    - Adjusts figure size dynamically to fit all layers properly.
    - Resizes PE units if the layout exceeds a threshold size.
    """

    fabric_width, fabric_height, offset_x, offset_y = fabric_dimensions
    
    base_size = 0.5  # Adjust this value for better visualization
    fig_width = max(10, total_pe_cols * base_size)
    fig_height = max(6, total_pe_rows * base_size)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    ax.set_xlim(0, fabric_width)   # X increases right → PE Columns
    ax.set_ylim(fabric_height, 0)  # Y increases downward → PE Rows

    total_colors = len(layer_coordinates_map)
    colors = plt.cm.get_cmap('tab20', total_colors).colors
    
    # Track occupied PEs in a 2D grid
    occupied_grid = np.zeros((fabric_height, fabric_width), dtype=bool)

    # Draw layers with width & height
    for layer_index, coords in layer_coordinates_map.items():
        start_x = coords["layer_start_x"] + offset_x
        start_y = coords["layer_start_y"] + offset_y
        width = coords["layer_pe_cols"]
        height = coords["layer_pe_rows"]

        # Mark occupied area
        occupied_grid[start_y:start_y+height, start_x:start_x+width] = True

        # Get data shapes
        is_layer_in_amg = layer_params_map.get(layer_index, None)
        if is_layer_in_amg is not None:
            data_shapes = is_layer_in_amg["layer_data_shapes"]
            shape_text = f"A({data_shapes['layer_M']}x{data_shapes['layer_N']})"
        else:
            shape_text = "A(0x0)"

        # Create rectangle for the layer
        rect = patches.Rectangle((start_x, start_y), width, height, 
                                 linewidth=2, edgecolor='black', facecolor=colors[layer_index % len(colors)], alpha=0.6)
        ax.add_patch(rect)

        # Add text inside the layer for data shape
        ax.text(start_x + width / 2, start_y + height / 2, f"L{layer_index+1}\n{shape_text}", 
                fontsize=10, ha='center', va='center', fontweight='bold', color='black')

        # Add width & height labels at the edges
        ax.text(start_x + width / 2, start_y - 0.3, f"{width}", 
                fontsize=9, ha='center', va='bottom', fontweight='bold', color='black')

        ax.text(start_x - 0.3, start_y + height / 2, f"{height}", 
                fontsize=9, ha='right', va='center', fontweight='bold', color='black')


    for y in range(offset_y, offset_y + total_pe_rows):
        for x in range(offset_x, offset_x + total_pe_cols):
            if not occupied_grid[y, x]:  # If this PE is not occupied
                empty_rect = patches.Rectangle((x, y), 1, 1, 
                                               linewidth=0.5, edgecolor='gray', linestyle='dashed', 
                                               facecolor='lightgray', alpha=0.3)
                ax.add_patch(empty_rect)
                ax.text(x + 0.5, y + 0.5, "Empty", 
                        fontsize=6, ha='center', va='center', color='gray')

    ax.set_xticks(range(0, fabric_width + 1, 1))  # PE Columns
    ax.set_yticks(range(0, fabric_height + 1, 1)) # PE Rows
    ax.grid(True, linestyle='--', linewidth=0.5)
    ax.set_title(f"Fabric Layout with Empty Regions Inside @set_rectangle({total_pe_cols}, {total_pe_rows})")
    ax.set_xlabel("PE Columns")
    ax.set_ylabel("PE Rows")

    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
