import numpy as np
from scipy.sparse import diags, kron, eye

def build_A_2d(n):
    main = 2.0 * np.ones(n)
    off  = -1.0 * np.ones(n - 1)
    L1 = diags([off, main, off], offsets=[-1, 0, 1], shape=(n, n))
    Ix = eye(n)
    Iy = eye(n)
    A = kron(Iy, L1) + kron(L1, Ix)
    
    # Visualize the sparse matrix
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 6))
    plt.spy(A, markersize=1)
    plt.title(f"Sparse matrix A (n={n})")
    plt.xlabel("Column")
    plt.ylabel("Row")
    plt.savefig("sparse_matrix_visualization.png", dpi=300)
    plt.show()
    
    return A

def apply_5pt_matrix_free(u, h):
  
    # Print vector values in a box
    print("Vector up (padded):")
    print("┌" + "─" * (u.shape[1] * 8 - 1) + "┐")
    for i in range(u.shape[0]):
        row_str = "│"
        for j in range(u.shape[1]):
            row_str += f"{u[i,j]:7.3f} "
        row_str = row_str.rstrip() + "│"
        print(row_str)
    print("└" + "─" * (u.shape[1] * 8 - 1) + "┘")
    print()

    up = np.pad(u, 1, mode='constant', constant_values=0.0)
    
    # Show 5-point stencil pattern
    print("5-point stencil pattern:")
    print("  ┌─┐")
    print("  │N│")
    print("┌─┼─┼─┐")
    print("│W│C│E│")
    print("└─┼─┼─┘")
    print("  │S│")
    print("  └─┘")
    print("Where: C=center, N=north, S=south, E=east, W=west")
    print("Formula: 4*C - (N+S+E+W)")
    print()
    
    y = (4.0 * up[1:-1, 1:-1] - (up[2:, 1:-1] + up[:-2, 1:-1] + up[1:-1, 2:] + up[1:-1, :-2])) / (h*h)
    
    return y

def test_2d(n=8):
    h = 1.0 / (n + 1)
    u = np.random.default_rng(0).standard_normal((n, n))
    y_stencil = apply_5pt_matrix_free(u, h)
    print("Applied stencil to unknown vector:")

    A = build_A_2d(n) / (h*h)
    y_matvec  = (A @ u.ravel()).reshape(n, n)
    print("Applied Matrix into unknown vector:")
    return float(np.max(np.abs(y_stencil - y_matvec)))

print("Running equality checks (stencil == assembled SpMV)...")
print(f"2D (n=10) max abs diff: {test_2d(10):.3e}")

