import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt  # Import for colormap


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

def visualize_layout_with_empty_plotly(fabric_dimensions, layer_coordinates_map, layer_params_map, total_pe_cols, total_pe_rows, filename="fabric_layout_with_empty_plotly.png"):
    """
    Visualizes the layout of layers on the full fabric grid with visible grid lines and specified colors using Plotly.
    Includes the area defined by total_pe_cols and total_pe_rows.
    """

    fabric_width, fabric_height, offset_x, offset_y = fabric_dimensions

    # Initialize figure for Plotly
    fig = go.Figure()

    # Define specific colors for the layers
    layer_colors = ["rgba(255, 182, 193, 0.7)", "rgba(173, 216, 230, 0.7)", "rgba(144, 238, 144, 0.7)", "rgba(255, 255, 0, 0.7)"] # LightPink, LightBlue, LightGreen, Yellow

    # Track occupied PEs in a 2D grid (now based on total dimensions)
    occupied_grid = np.zeros((total_pe_rows, total_pe_cols), dtype=bool)

    # Draw layers with width & height
    for layer_index, coords in layer_coordinates_map.items():
        start_x = coords["layer_start_x"]
        start_y = coords["layer_start_y"]
        width = coords["layer_pe_cols"]
        height = coords["layer_pe_rows"]

        # Mark occupied area (using the provided coordinates directly)
        occupied_grid[start_y:start_y+height, start_x:start_x+width] = True

        # Get data shapes
        is_layer_in_amg = layer_params_map.get(layer_index, None)
        if is_layer_in_amg is not None:
            data_shapes = is_layer_in_amg["layer_data_shapes"]
            shape_text = f"A({data_shapes['layer_M']}x{data_shapes['layer_N']})"
        else:
            shape_text = "A(0x0)"

        # Get color for the layer
        color = layer_colors[layer_index % len(layer_colors)]

        # Add rectangle for the layer
        fig.add_shape(
            type="rect",
            x0=start_x,
            y0=start_y,
            x1=start_x + width,
            y1=start_y + height,
            line=dict(color="black", width=1),
            fillcolor=color,
            opacity=1
        )

        # Add text inside the layer for data shape
        fig.add_annotation(
            x=start_x + width / 2,
            y=start_y + height / 2,
            text=f"L{layer_index + 1}<br>{shape_text}",
            showarrow=False,
            font=dict(size=10, color="black", weight='bold'),
            align="center"
        )

        # Add width & height labels at the edges
        fig.add_annotation(
            x=start_x + width / 2,
            y=start_y - 0.3,
            text=str(width),
            showarrow=False,
            font=dict(size=8, color="black", weight='bold'),
            align="center",
            yshift=-8
        )

        fig.add_annotation(
            x=start_x - 0.3,
            y=start_y + height / 2,
            text=str(height),
            showarrow=False,
            font=dict(size=8, color="black", weight='bold'),
            align="right",
            xshift=-8
        )

    # Draw the full PE grid and mark empty spaces
    for y in range(total_pe_rows):
        for x in range(total_pe_cols):
            fig.add_shape(
                type="rect",
                x0=x,
                y0=y,
                x1=x + 1,
                y1=y + 1,
                line=dict(color="lightgray", width=0.5, dash="dot"),
                fillcolor="white" if not occupied_grid[y, x] else None, # Only fill white if not occupied
                opacity=0.5 if not occupied_grid[y, x] else 0
            )
            if not occupied_grid[y, x]:
                fig.add_annotation(
                    x=x + 0.5,
                    y=y + 0.5,
                    text="Empty",
                    showarrow=False,
                    font=dict(size=6, color="gray"),
                    align="center"
                )

    # Update layout for fixed size and visible grid
    fig.update_layout(
        title=f"Fabric Layout with Empty Regions Inside @set_rectangle({total_pe_cols}, {total_pe_rows})",
        xaxis=dict(title="PE Columns", tickmode="linear", tick0=0, dtick=1, showgrid=True, gridcolor="lightgray", zeroline=False, range=[0, total_pe_cols]),
        yaxis=dict(title="PE Rows", tickmode="linear", tick0=0, dtick=1, showgrid=True, gridcolor="lightgray", zeroline=False, range=[total_pe_rows, 0]),
        width=800,
        height=600,
        showlegend=False,
        plot_bgcolor="white",
        template="plotly_white"
    )

    # Save as HTML
    # fig.write_html(filename)
    fig.write_image(filename)
    print(f"Saved layout image to {filename}")
    fig.show()
