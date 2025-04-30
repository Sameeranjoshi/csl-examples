from tabulate import tabulate
import numpy as np
from scipy import sparse as sp
import scipy.sparse.linalg as spla
from pyamg.aggregation.aggregation import smoothed_aggregation_solver
import utilities as ut
from time_utils import OperatorTiming
import time

def each_layer_solver_down(level, x_level, b_level, ml, setup_config, level_id, hostprofiling):
    """Handle downward pass operations for a single level with timing"""
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
    print("smooth X", x)
    # Time residual calculation
    residual_start = time.time()
    r = b - A @ x   # residual
    timing.residual_time = time.time() - residual_start
    
    # Time restriction
    restrict_start = time.time()
    b_coarse = R @ r    # restrict
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


def print_table_shapes(levels):
    level_data = []
    for idx, level in enumerate(levels):
        level_info = {'level': idx}
        # Handle A
        if hasattr(level, 'A') and level.A is not None:
            format_str = level.A.format if hasattr(level.A, 'format') else 'dense'
            level_info[f'A_shape({level.A.dtype},{format_str})'] = f"{level.A.shape}"
        else:
            level_info['A_shape'] = "N/A"

        # Handle B
        if hasattr(level, 'B') and level.B is not None:
            format_str = level.B.format if hasattr(level.B, 'format') else 'dense'
            level_info[f'B_shape({level.B.dtype},{format_str})'] = f"{level.B.shape}"

        # Handle P
        if hasattr(level, 'P') and level.P is not None:
            format_str = level.P.format if hasattr(level.P, 'format') else 'dense'
            level_info[f'P_shape({level.P.dtype},{format_str})'] = f"{level.P.shape}"

        # Handle R
        if hasattr(level, 'R') and level.R is not None:
            format_str = level.R.format if hasattr(level.R, 'format') else 'dense'
            level_info[f'R_shape({level.R.dtype},{format_str})'] = f"{level.R.shape}"

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


#############################################

def smooth_aggregate_setup_only(A, x, b, solver, max_level=None, max_coarse=None):
    coarse_solver_callable = solver
    smoothed_aggregation_solver_config = {
        'B': b,
        'symmetry': 'symmetric',
        'aggregate': ('lloyd', {'ratio': 0.10}),  # Reduce coarsening aggressiveness
        'strength': ('symmetric', {'theta': 0.10}),  # Capture more connections
        'smooth': 'jacobi',
        # withrho is essential for answers to be correct
        'presmoother': ('jacobi', {'omega': 1.0/3.0, 'iterations': 5, 'withrho': False}),
        'postsmoother': ('jacobi', {'omega': 1.0/3.0, 'iterations': 5, 'withrho': False}),  
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
