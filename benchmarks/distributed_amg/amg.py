
import warnings
from matplotlib import pyplot as plt
from tabulate import tabulate
import numpy as np
from scipy import sparse as sp
import scipy.sparse.linalg as spla
import pyamg    # Maybe we need this, try if it's modular?
import pyamg.relaxation.relaxation as pyamg_smoother
from pyamg.classical.classical import ruge_stuben_solver
from pyamg.aggregation.aggregation import smoothed_aggregation_solver
from pyamg import smoothed_aggregation_solver, ruge_stuben_solver
from scipy.sparse import random
import math
import utilities as ut
from time_utils import OperatorTiming
import time
from typing import Dict, Optional, Tuple, Any

def each_layer_solver_down(level, x_level, b_level, ml, setup_config, level_id, hostprofiling):
    """Handle downward pass operations for a single level with timing"""
    print("level_id", level_id)
    timing = OperatorTiming()
    level_start = time.time()
    
    A, R = level.A, level.R
    # print("A format, A dtype:", A.format, A.dtype)
    # print("R format, R dtype:", R.format, R.dtype)
    x = x_level[level_id]
    b = b_level[level_id]
    
    # Time pre-smoothing
    smooth_start = time.time()
    ut.jacobi_csr(A, x, b, 
                 omega=get_omega_from_presmoother(setup_config), 
                 iterations=get_iterations_from_presmoother(setup_config))
    timing.smoothing_time = time.time() - smooth_start
    # print("smooth X", x.flatten())
    # Time residual calculation
    residual_start = time.time()
    r = b - A @ x   # residual
    # print("b", b.flatten())
    # print("A", A.toarray())
    # print("x", x.flatten())
    # print("r", r.flatten())
    timing.residual_time = time.time() - residual_start
    
    # Time restriction
    restrict_start = time.time()
    b_coarse = R @ r    # restrict
    # print("b_coarse", b_coarse.flatten())
    timing.restriction_time = time.time() - restrict_start
    
    x_coarse = np.zeros_like(b_coarse)
    
    timing.total_time = time.time() - level_start
    
    # Add timing to profiler
    hostprofiling.add_timing(level_id, timing, "down")
    
    return b_coarse, x_coarse, timing

def each_layer_solver_up(level, x_level, b_level, ml, setup_config, level_id, x_lower_level, hostprofiling):
    """Handle upward pass operations for a single level with timing"""
    timing = OperatorTiming()
    level_start = time.time()
    
    A, P = level.A, level.P
    # print("A format UP, A dtype:", A.format, A.dtype)
    # print("P format UP, P dtype:", P.format, P.dtype)
    x = x_level[level_id]
    b = b_level[level_id]

    # Time prolongation
    prolong_start = time.time()
    x += P @ x_lower_level  # Prolongation
    print("level_id", level_id)
    print("x+= P @ x_lower_level = ", x.flatten())
    timing.prolongation_time = time.time() - prolong_start
    
    # Time post-smoothing
    smooth_start = time.time()
    ut.jacobi_csr(A, x, b,
                 omega=get_omega_from_postsmoother(setup_config),
                 iterations=get_iterations_from_postsmoother(setup_config))
    timing.smoothing_time = time.time() - smooth_start
    
    timing.total_time = time.time() - level_start
    
    # Add timing to profiler
    hostprofiling.add_timing(level_id, timing, "up")
    
    return x, timing

# only V cycle, iterative version
def AMG_test(ml, setup_config, x_level, b_level, solver, max_level=10, max_coarse=2, max_ite=1):    
    # if only 1 level solve directly
    if not ml.levels:
        raise RuntimeError("AMG setup failed: No levels generated.")

    level_data_clone = []
    coarse_data = []

    for _ in range(max_ite):  # Run V-cycle up to max_iter or until convergence
        # solve phase layer by layer.
        for i, level in enumerate(ml.levels[:-1]):
            A, P, R = level.A, level.P, level.R
            x = x_level[i]
            b = b_level[i]

            # level.presmoother(A, x, b)
            ut.jacobi_csr(A, x, b, omega=get_omega_from_postsmoother(setup_config), iterations=get_iterations_from_postsmoother(setup_config))            
            r = b - A @ x
            b_coarse = R @ r
            x_coarse = np.zeros_like(b_coarse)
            
            b_level[i + 1] = b_coarse  # Directly store in the next level
            x_level[i + 1] = x_coarse
            
        b_coarsest = b_level[-1]
        x_coarsest = x_level[-1]
        A_coarsest = ml.levels[-1].A
        if isinstance(solver, tuple):
            solver_fn, solver_kwargs = solver
            sol_result = solver_fn(A_coarsest, b_coarsest, **solver_kwargs)  # Pass additional arguments
            x_coarsest[:] = sol_result[0] if isinstance(sol_result, tuple) else sol_result
        else:
            x_coarsest[:] = solver(A_coarsest, b_coarsest)  # Call without extra kwargs
        print("After coarse solver:")
        debugprint(ml.levels, b_level, x_level)
    
        # Solve up
        for i in reversed(range(len(ml.levels) - 1)):            
            P = ml.levels[i].P
            x_level[i] += P @ x_level[i+1]  # Prolongation

            # Apply post-smoother
            # ml.levels[i].postsmoother(ml.levels[i].A, x_level[i], b_level[i])
            ut.jacobi_csr(ml.levels[i].A, x_level[i], b_level[i], omega=get_omega_from_postsmoother(setup_config), iterations=get_iterations_from_postsmoother(setup_config))
    
    
    print("After REF final solver:")
    debugprint(ml.levels, b_level, x_level)
    return b_level[0], x_level[0]
    
# solve a linear system A * x = b
# where A is a symmetric positive definite matrix
# The AMG algorithm is from pyamg, https://github.com/pyamg/pyamg

# only V cycle, iterative version
def AMG_only_solve(A_csr, b0, x0, tol, max_ite, max_levels, max_coarse, solver):
    """HAND WRITTEN AMG ALGORITHM FROM SC AMGT PAPER, MATRIX COMPUTATIONS BOOK AND WIKIPEDIA"""

    # Start as of now with the solve phase use setup from pyamg.
    ml, setup_config = smooth_aggregate_setup_only(A_csr, x0, b0, solver, max_levels, max_coarse)
    # visualize(ml)
        
    # print(ml)
    # print_table_shapes(ml.levels)
    # print_table_data(ml.levels)
    
    # if only 1 level solve directly
    if not ml.levels:
        raise RuntimeError("AMG setup failed: No levels generated.")

    level_data_clone = []
    coarse_data = []
    V_levels = len(ml.levels)
    x_level = [None] * V_levels
    b_level = [None] * V_levels

    x_level[0] = np.copy(x0)
    b_level[0] = np.copy(b0)

    for _ in range(max_ite):  # Run V-cycle up to max_iter or until convergence
        if V_levels == 1:
            A = ml.levels[0].A
            x = x_level[0]
            b = b_level[0]
            # x, rho = scipy_iterative_solver(A_csr, b0, x0, tol, max_ite=max_ite)
            if isinstance(solver, tuple):
                solver_fn, solver_kwargs = solver
                x[:] = solver_fn(A, b, **solver_kwargs)  # Pass additional arguments
            else:
                solver_fn = solver  # Direct function call
                x[:] = solver_fn(A, b)  # Call without extra kwargs
        else:   # N layers   
            # solve phase layer by layer.
            for i, level in enumerate(ml.levels[:-1]):
                A, P, R = level.A, level.P, level.R
                x = x_level[i]
                b = b_level[i]
                # level.presmoother(A, x, b)
                ut.jacobi_csr(A, x, b, omega=get_omega_from_presmoother(setup_config), iterations=get_iterations_from_presmoother(setup_config))
                r = b - A @ x
                b_coarse = R @ r
                
                b_level[i + 1] = b_coarse  # Directly store in the next level
                x_coarse = np.zeros_like(b_coarse)
                x_level[i + 1] = x_coarse
        
            # print after solve down.
            # debugprint(ml.levels, b_level, x_level)
            # solve coarse
            b_coarsest = b_level[-1]
            x_coarsest = x_level[-1]
            A_coarsest = ml.levels[-1].A
            if isinstance(solver, tuple):
                solver_fn, solver_kwargs = solver
                sol_result = solver_fn(A_coarsest, b_coarsest, **solver_kwargs)  # Pass additional arguments
                x_coarsest[:] = sol_result[0] if isinstance(sol_result, tuple) else sol_result
            else:
                x_coarsest[:] = solver(A_coarsest, b_coarsest)  # Call without extra kwargs
        
            # Solve up
            for i in reversed(range(len(ml.levels) - 1)):
                P = ml.levels[i].P
                x_level[i] += P @ x_level[i+1]  # Prolongation

                # Apply post-smoother
                # ml.levels[i].postsmoother(ml.levels[i].A, x_level[i], b_level[i])
                ut.jacobi_csr(ml.levels[i].A, x_level[i], b_level[i], omega=get_omega_from_presmoother(setup_config), iterations=get_iterations_from_presmoother(setup_config))

        resi = b_level[0] - ml.levels[0].A @ x_level[0]
        relative_residual = np.linalg.norm(resi)
        if relative_residual < tol:
            print(f"Inside custom-solve: Relative residual: {relative_residual}, Tolerance: {tol}, Converged at iteration {_}")
            break

    rho = np.dot(resi, resi)
    return x_level[0], rho

def print_table_shapes(levels):
    level_data = []
    for idx, level in enumerate(levels):
        level_info = {'level': idx}
        # Handle A
        if hasattr(level, 'A') and level.A is not None:
            # format_str = level.A.format if hasattr(level.A, 'format') else 'dense'
            # level_info[f'A_shape({level.A.dtype},{format_str})'] = f"{level.A.shape}"
            level_info[f'A_shape({level.A.dtype})'] = f"{level.A.shape}"
        else:
            level_info['A_shape'] = "N/A"

        # # Handle B
        # if hasattr(level, 'B') and level.B is not None:
        #     format_str = level.B.format if hasattr(level.B, 'format') else 'dense'
        #     level_info[f'B_shape({level.B.dtype},{format_str})'] = f"{level.B.shape}"

        # Handle R
        if hasattr(level, 'R') and level.R is not None:
            format_str = level.R.format if hasattr(level.R, 'format') else 'dense'
            level_info[f'R_shape({level.R.dtype},{format_str})'] = f"{level.R.shape}"

        # Handle P
        if hasattr(level, 'P') and level.P is not None:
            format_str = level.P.format if hasattr(level.P, 'format') else 'dense'
            level_info[f'P_shape({level.P.dtype},{format_str})'] = f"{level.P.shape}"

        # Handle presmoother
        if hasattr(level, 'presmoother') and level.presmoother:
            level_info['presmoother'] = (
                level.presmoother.func.__name__ if callable(level.presmoother) else str(level.presmoother)
            )

        # Handle postsmoother
        if hasattr(level, 'postsmoother') and level.postsmoother:
            level_info['postsmoother'] = (
                level.postsmoother.func.__name__ if callable(level.postsmoother) else str(level.postsmoother)
            )

        level_data.append(level_info)

    # Print table
    print(tabulate(level_data, headers="keys", tablefmt="grid"))

def print_table_data(levels):
    level_data = []
    
    for idx, level in enumerate(levels):
        A, b = level.A, level.B

        # Convert first row of A to a readable format
        A_str = np.array2string(A.toarray()[0, 0:5], separator=', ')

        # Convert first 5 elements of B to a readable format
        B_str = np.array2string(b[0:5].flatten(), separator=', ')


        level_data.append({
            "Level": idx,
            "A[0:5]": A_str,
            "B[0:5]": B_str,
        })

    print(tabulate(level_data, headers="keys", tablefmt="grid"))

def debugprint(levels, b_level, x_level):
    level_data_clone = []
    for i, level in enumerate(levels):
        A, b = level.A, level.B
        
        level_data_clone.append({
            "level": i,
            "A": A.toarray()[0,0:10] if hasattr(A, 'toarray') else A[0,0:10],
            # "B_original": b[0:10].flatten(), // What values are these?
            "b_computed": b_level[i][0:10].flatten() if b_level[i] is not None else "--",
            "x_computed": x_level[i][0:10].flatten() if x_level[i] is not None else "--",
        })
    print(tabulate(level_data_clone, headers="keys", tablefmt="grid"))  

def visualize(ml):
    # Determine the number of levels
    num_levels = len(ml.levels)
    
    # Calculate the number of rows and columns for the grid
    grid_size = math.ceil(math.sqrt(num_levels))  # Calculate grid size (square-like)
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(5 * grid_size, 5 * grid_size))
    
    # Flatten the axes array in case it's a 2D array (for easy indexing)
    axes = axes.flatten()

    for i, level in enumerate(ml.levels):
        A_viz = level.A
        R_viz = level.R if hasattr(level, 'R') else None
        P_viz = level.P if hasattr(level, 'P') else None
        # Plot sparsity pattern for A
        axes[i].spy(A_viz.toarray(), markersize=2)
        axes[i].set_title(f"Level {i+1} - A")
        # Plot sparsity pattern for R
        if R_viz is not None and i+1 < len(axes):
            axes[i+1].spy(R_viz.toarray(), markersize=2)
            axes[i+1].set_title(f"Level {i+1} - R")
        # Plot sparsity pattern for P
        if P_viz is not None and i+2 < len(axes):
            axes[i+2].spy(P_viz.toarray(), markersize=2)
            axes[i+2].set_title(f"Level {i+1} - P")
    
    # Hide any unused subplots (if num_levels is less than grid_size^2)
    for j in range(num_levels, len(axes)):
        axes[j].axis('off')

    # Adjust layout and save the figure
    plt.tight_layout()
    plt.savefig("sparse_matrix_visualization.png", dpi=300)
    plt.show()
    
def get_omega_from_postsmoother(smoothed_aggregation_solver_config):
    # Extract omega from postsmoother configuration
    postsmoother_config = smoothed_aggregation_solver_config.get('postsmoother', None)
    if postsmoother_config and isinstance(postsmoother_config, tuple):
        postsmoother_params = postsmoother_config[1]
        omega = postsmoother_params.get('omega', None)
        return omega 
    else:
        print("Postsmoother omega not found in configuration")
        exit(1)

def get_iterations_from_postsmoother(smoothed_aggregation_solver_config):
    # Extract iterations from postsmoother configuration
    postsmoother_config = smoothed_aggregation_solver_config.get('postsmoother', None)
    if postsmoother_config and isinstance(postsmoother_config, tuple):
        postsmoother_params = postsmoother_config[1]
        iterations = postsmoother_params.get('iterations', None)
        # iterations has to be greater than 0.
        if iterations is not None and iterations <= 0:
            print("Postsmoother iterations must be greater than 0")
            exit(1)
        return iterations
    else:
        print("Postsmoother iterations not found in configuration")
        exit(1)
        
def get_omega_from_presmoother(smoothed_aggregation_solver_config):
    # Extract omega from presmoother configuration
    presmoother_config = smoothed_aggregation_solver_config.get('presmoother', None)
    if presmoother_config and isinstance(presmoother_config, tuple):
        presmoother_params = presmoother_config[1]
        omega = presmoother_params.get('omega', None)
        return omega 
    else:
        print("Presmoother omega not found in configuration")
        exit(1)

def get_iterations_from_presmoother(smoothed_aggregation_solver_config):
    # Extract iterations from presmoother configuration
    presmoother_config = smoothed_aggregation_solver_config.get('presmoother', None)
    if presmoother_config and isinstance(presmoother_config, tuple):
        presmoother_params = presmoother_config[1]
        iterations = presmoother_params.get('iterations', None)
        # iterations has to be greater than 0.
        if iterations is not None and iterations <= 0:
            print("Presmoother iterations must be greater than 0")
            exit(1)
        return iterations
    else:
        print("Presmoother iterations not found in configuration")
        exit(1)
#############################################
def rs_base(A, x, b, max_iter=100, solver='cg'):
    np.random.seed(100)
    ml = ruge_stuben_solver(A, coarse_solver=solver)  # Multi-level solver
    print(ml)
    x_pyamg = ml.solve(b, x0=x, maxiter=max_iter)
    return x_pyamg

def smooth_aggregate_base(A, x, b, max_iter=100, solver='cg'):
    np.random.seed(100)  
    ml = smoothed_aggregation_solver(A, B=b, coarse_solver=solver)
    print(ml)
    x_sol = ml.solve(b, x0=x, maxiter=max_iter)
    return x_sol

def scipy_direct_solver(A, b):
    coarse_data = []
    x = spla.spsolve(A, b)
    A_str = np.array2string(A.toarray()[0, 0:5], separator=', ')
    B_str = np.array2string(b[0:5].flatten(), separator=', ')
    X_str = np.array2string(x[0:5].flatten(), separator=', ')
    # coarse_data.append({
    #     "Level": 0,
    #     "A_coarse[0:5]": A_str,
    #     "B_coarse[0:5]": B_str,
    #     "X_solved[0:5]": X_str,
    # })
    # print(tabulate(coarse_data, headers="keys", tablefmt="grid"))
    r = b - A.dot(x)
    rho = np.dot(r, r)
    return x

# TODO: Fix the atol and issues like those.
def scipy_iterative_solver(A, b, x):
    x, converged = spla.cg(A, b, x0=x)
    if converged > 0:
        print("spla.cg from scipy_iterative_solver did not converge, maybe use more iterations or reduce tolerance.")
    
    # calculate residual
    r = b - A.dot(x)
    rho = np.dot(r, r)
    return x

#############################################
def rs_modified(A, x, b, max_ite, relative_tol, solver, max_level=None, max_coarse=None):
 
      ml, setup_config = rs_setup_only(A, x, b, solver, max_level, max_coarse)
      print(ml)
      # 4. solve
      residual = []
      solve_config = {
            'maxiter': max_ite, # TODO: Full solver=max_ite, as_preconditioner=1, only_one_V_cycle for testing.
            'residuals': residual,
            'tol': relative_tol,
            'return_info': True,
      }
      x_sol, info = ml.solve(b, x0=x, **solve_config)
      if info != 0:
         warnings.warn(
             "\033[91mPyAMG did not converge, this is because I have explicitly set the max_iter=1, "
             "this means only do 1 cycle of AMG, as a full solver it should keep repeating the cycles until max_iter, "
             "now as a preconditioner cycles=1 followed by CG/GMRES/...: \033[0m" + str(info), 
             UserWarning
         )

      return x_sol

def smooth_aggregate_modified(A, x, b, max_ite, relative_tol, solver, max_level=None, max_coarse=None):

    ml, setup_config = smooth_aggregate_setup_only(A, x, b, solver, max_level, max_coarse)
    print(ml)
    # print_table_shapes(ml.levels)    

    residual = []
    solve_config = {
        'maxiter': max_ite,
        'residuals': residual,
        'tol': relative_tol,
        'return_info': True,
    }
    # create a callback method which prints the rho and tol at each iteration
    x_sol, info = ml.solve(b, x0=x, **solve_config)
    if info != 0:
        warnings.warn(
            "\033[91mPyAMG did not converge, this is because I have explicitly set the max_iter=1, "
            "this means only do 1 cycle of AMG, as a full solver it should keep repeating the cycles until max_iter, "
            "now as a preconditioner cycles=1 followed by CG/GMRES/...: \033[0m" + str(info), 
            UserWarning
        )

    # for i, r in enumerate(residual):
    #     print(f"[SA_modified]Iteration {i}: Residual {r}")
    
    # print_table_data(ml.levels)
    return x_sol
      
def smooth_aggregate_setup_only(A, x, b, solver, max_level=None, max_coarse=None):
    coarse_solver_callable = solver
    smoothed_aggregation_solver_config = {
        'B': b,
        'symmetry': 'symmetric',
        'aggregate': ('lloyd', {'ratio': 0.50}),  # increase for more layers
        'strength': ('symmetric', {'theta': 0.50}),  # increase for more layers
        'smooth': 'jacobi',
        # withrho is essential for answers to be correct
        'presmoother': ('jacobi', {'omega': 1.0/3.0, 'iterations': 2, 'withrho': False}),
        'postsmoother': ('jacobi', {'omega': 1.0/3.0, 'iterations': 2, 'withrho': False}),  
        'improve_candidates': (('gauss_seidel', {'sweep': 'symmetric', 'iterations': 6}), None),
        'coarse_solver': coarse_solver_callable
    }
    if max_coarse is not None:
        smoothed_aggregation_solver_config['max_coarse'] = max_coarse
    if max_level is not None:
        smoothed_aggregation_solver_config['max_levels'] = max_level

    # 3. solver
    np.random.seed(100)  
    # This is essential to pass seed as smooth is non-deterministic, 
    # https://github.com/pyamg/pyamg/issues/350
    ml = smoothed_aggregation_solver(A, **smoothed_aggregation_solver_config)

    return ml, smoothed_aggregation_solver_config

def rs_setup_only(A, x, b, solver, max_level=None, max_coarse=None):
    
    coarse_solver_callable = solver
    ruge_stuben_config = {
        'strength': ('classical', {'theta': 0.4}),  # Capture more connections
        'CF': ('RS', {'second_pass': True}),  # Standard RS coarsening
        'interpolation': 'classical',  # Slowest interpolation method
        'presmoother': ('jacobi', {'omega': 1.0/3.0, 'iterations': 5, 'withrho': False}),
        'postsmoother': ('jacobi', {'omega': 1.0/3.0, 'iterations': 5, 'withrho': False}),
        'coarse_solver': coarse_solver_callable  
    }
    if max_coarse is not None:
        ruge_stuben_config['max_coarse'] = max_coarse
    if max_level is not None:
        ruge_stuben_config['max_levels'] = max_level

    np.random.seed(100)  
    ml = ruge_stuben_solver(A, **ruge_stuben_config)

    return ml, ruge_stuben_config
