# others
import warnings
from tabulate import tabulate
# numpy
import numpy as np
from numpy import linalg as LA
from numpy.testing import TestCase, assert_equal, assert_almost_equal, \
    assert_array_almost_equal
# scipy
from scipy import sparse as sp
import scipy.sparse.linalg as spla
import scipy.sparse.linalg as spla
from scipy.sparse import csr_matrix, coo_matrix, SparseEfficiencyWarning, isspmatrix_csr
# pyamg
import pyamg    # Maybe we need this, try if it's modular?
import pyamg.relaxation.relaxation as pyamg_smoother
from pyamg.gallery import poisson, load_example
from pyamg.strength import classical_strength_of_connection
from pyamg.classical import split
from pyamg.classical.classical import ruge_stuben_solver
from pyamg.aggregation.aggregation import smoothed_aggregation_solver
from pyamg.classical.interpolate import direct_interpolation, \
    classical_interpolation

#others
from cg import conjugateGradient
from pyamg import smoothed_aggregation_solver, ruge_stuben_solver
from scipy.sparse import random


# solve a linear system A * x = b
# where A is a symmetric positive definite matrix
# The AMG algorithm is from pyamg, https://github.com/pyamg/pyamg

# only V cycle, iterative version
def AMG_only_solve(A_csr, b0, x0, tol, max_ite, max_levels, max_coarse, solver):
    """HAND WRITTEN AMG ALGORITHM FROM SC AMGT PAPER, MATRIX COMPUTATIONS BOOK AND WIKIPEDIA"""
    # 1. setup
    # 2. V cycle
    # 3. return x, rho
    
    # # do some checks
    # if not isspmatrix_csr(A_csr):
    #     raise TypeError("Matrix A must be in CSR format")
    # if A_csr.shape[0] != A_csr.shape[1]:
    #     raise ValueError("Matrix A must be square")
    # if A_csr.shape[0] != b0.shape[0]:
    #     raise ValueError("Matrix A and vector b must have the same number of rows")
    
    # # 1. setup
    # # copy along with type
    # x = np.copy(x0)
    # b = np.copy(b0)
    # levels = []
    
    # levels[0].A = A_csr

    # i = 0
    # while len(levels) < max_levels and levels[i].A.shape[0] > max_coarse:
    #     # If we are at level 0, set up initial values for x and b
    #     if i == 0:
    #         levels[i].x = x
    #         levels[i].b = b

    #     # Compute the interpolation matrix (prolongation operator)
    #     levels[i].R = restriction(i, levels[i].A, splitting, C_F_splitting)
    #     levels[i].P = levels[i].R.T
    #     levels[i + 1].A = levels[i].R @ levels[i].A @ levels[i].P

    #     # Move to the next level
    #     i = i + 1

    # # print levels
    # print_table_shapes(levels)
    
    
    # Start as of now with the solve phase use setup from pyamg.
    ml, setup_config = smooth_aggregate_setup_only(A_csr, x0, b0, solver, max_levels, max_coarse)
    print(ml)
    # print_table_shapes(ml.levels)
    # print_table_data(ml.levels)
    
    # if only 1 level solve directly
    if not ml.levels:
        raise RuntimeError("AMG setup failed: No levels generated.")
    if len(ml.levels) == 1:
        # x, rho = scipy_iterative_solver(A_csr, x0, b0, tol, max_ite=max_ite)
        if isinstance(solver, tuple):
            solver_fn, solver_kwargs = solver
            x = solver_fn(A_csr, b0, **solver_kwargs)  # Pass additional arguments
        else:
            solver_fn = solver  # Direct function call
            x = solver_fn(A_csr, b0)  # Call without extra kwargs

        r = b0 - A_csr @ x
        rho = np.dot(r, r)
        return x, rho
    
    
    level_data_clone = []
    coarse_data = []
    x_level = [np.copy(x0)]
    b_level = [np.copy(b0)]
    
    # solve phase layer by layer.
    for i, level in enumerate(ml.levels[:-1]):
        A, b, P, R = level.A, level.B, level.P, level.R
        x = x_level[i]
        b = b_level[i]
        
        level.presmoother(A, x, b)
        r = b - A @ x
        b_coarse = R @ r
        
        b_level.append(b_coarse)
        x_coarse = np.zeros_like(b_coarse)
        x_level.append(x_coarse)     
    # print after solve down.
    # debugprint(ml.levels, b_level, x_level)
    # solve coarse
    b_coarsest = b_level[-1]
    x_coarsest = x_level[-1]
    A_coarsest = ml.levels[-1].A
    if isinstance(solver, tuple):
        solver_fn, solver_kwargs = solver
        x_coarsest[:] = solver_fn(A_coarsest, b_coarsest, **solver_kwargs)  # Pass additional arguments
    else:
        solver_fn = solver  # Direct function call
        x_coarsest[:] = solver_fn(A_coarsest, b_coarsest)  # Call without extra kwargs
    
    # Solve up
    for i in reversed(range(len(ml.levels) - 1)):
        P = ml.levels[i].P
        x_level[i] += P @ x_level[i+1]  # Prolongation

        # Apply post-smoother
        ml.levels[i].postsmoother(ml.levels[i].A, x_level[i], b_level[i])


    # rho
    r = b_level[0] - ml.levels[0].A @ x_level[0]
    rho = np.dot(r, r)
    return x_level[0], rho
        
def print_table_shapes(levels):
    level_data = []
    for idx, level in enumerate(levels):
        level_info = {'level': idx}

        # Handle A
        if hasattr(level, 'A') and level.A is not None:
            level_info[f'A_shape({level.A.dtype})'] = f"{level.A.shape}"
        else:
            level_info['A_shape'] = "N/A"

        # Handle B
        if hasattr(level, 'B') and level.B is not None:
            level_info[f'B_shape({level.B.dtype})'] = f"{level.B.shape}"

        # Handle P
        if hasattr(level, 'P') and level.P is not None:
            level_info[f'P_shape({level.P.dtype})'] = f"{level.P.shape}"

        # Handle R
        if hasattr(level, 'R') and level.R is not None:
            level_info[f'R_shape({level.R.dtype})'] = f"{level.R.shape}"

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
            "A": A.toarray()[0,0:5],
            "B_original": b[0:5].flatten(),
            "b_computed": b_level[i][0:5].flatten(),
            "x_computed": x_level[i][0:5].flatten()
        })
    print(tabulate(level_data_clone, headers="keys", tablefmt="grid"))  
    
#############################################
def rs_base(A, x, b, solver='cg'):
    np.random.seed(100)
    ml = ruge_stuben_solver(A, coarse_solver=solver)  # Multi-level solver
    print(ml)
    x_pyamg = ml.solve(b, x0=x)
    return x_pyamg
def smooth_aggregate_base(A, x, b, solver='cg'):
    np.random.seed(100)  
    ml = smoothed_aggregation_solver(A, B=b, coarse_solver=solver)
    print(ml)
    x_sol = ml.solve(b, x0=x)
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

def scipy_iterative_solver(A, x, b, atol, max_ite):
    x, converged = spla.cg(A, b, x0=x, atol=atol, maxiter=max_ite)
    if converged > 0:
        print("spla.cg from scipy_iterative_solver did not converge, maybe use more iterations or reduce tolerance.")
    
    # calculate residual
    r = b - A.dot(x)
    rho = np.dot(r, r)
    return x, rho

#############################################
def rs_modified(A, x, b, max_ite, relative_tol, solver, max_level=None, max_coarse=None):
 
      ml, setup_config = rs_setup_only(A, x, b, solver, max_level, max_coarse)
      print(ml)
      # 4. solve
      residual = []
      solve_config = {
            'maxiter': max_ite,
            'residuals': residual,
            'tol': relative_tol,
            'return_info': True,
      }
      x_sol, info = ml.solve(b, x0=x, **solve_config)
      if info != 0:
          print("PyAMG did not converge, maybe try changing it's parameters. halted at iterations: ", info)
    
      for i, r in enumerate(residual):
         print(f"[RS_modified]Iteration {i}: Residual {r}")
    
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
    if info!=0:
        print("PyAMG did not converge, maybe try changing it's parameters. halted at iterations: ", info)

    # for i, r in enumerate(residual):
    #     print(f"[SA_modified]Iteration {i}: Residual {r}")
    
    # print_table_data(ml.levels)
    return x_sol
      
def smooth_aggregate_setup_only(A, x, b, solver, max_level=None, max_coarse=None):
    coarse_solver_callable = solver
    smoothed_aggregation_solver_config = {
        'B': b,
        'symmetry': 'symmetric',
        'aggregate': ('lloyd', {'ratio': 0.70}),  # Reduce coarsening aggressiveness
        'strength': ('symmetric', {'theta': 0.05}),  # Capture more connections
        'smooth': 'jacobi',
        'presmoother': ('jacobi', {'omega': 1.0/3.0, 'iterations': 5}),
        'postsmoother': ('jacobi', {'omega': 1.0/3.0, 'iterations': 5}),
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

    # print_table_data(ml.levels)
    # print(ml)
    # print_table_shapes(ml.levels)
    
    return ml, smoothed_aggregation_solver_config

def rs_setup_only(A, x, b, solver, max_level=None, max_coarse=None):
    
    coarse_solver_callable = solver
    ruge_stuben_config = {
        'strength': ('classical', {'theta': 0.4}),  # Capture more connections
        'CF': ('RS', {'second_pass': True}),  # Standard RS coarsening
        'interpolation': 'classical',  # Slowest interpolation method
        'presmoother': ('jacobi', {'omega': 1.0/3.0, 'iterations': 5}),
        'postsmoother': ('jacobi', {'omega': 1.0/3.0, 'iterations': 5}),
        'coarse_solver': coarse_solver_callable  
    }
    if max_coarse is not None:
        ruge_stuben_config['max_coarse'] = max_coarse
    if max_level is not None:
        ruge_stuben_config['max_levels'] = max_level

    np.random.seed(100)  
    ml = ruge_stuben_solver(A, **ruge_stuben_config)

    return ml, ruge_stuben_config

#########Visit later
class smoother:
    def __init__(self, level, x_input=None):
        self.A = level.level_A
        self.b = level.level_b
        self.x = x_input
        
    def are_sparse_matrices_equal(self, A, B):
        # Check if the shapes and the number of non-zero elements are the same
        if A.shape != B.shape or A.nnz != B.nnz:
            raise ValueError("Sparse matrices are not equal: shape or nnz mismatch")
        
        # Check if the data, indices, and indptr are the same
        if not (A.data == B.data).all() or not (A.indices == B.indices).all() or not (A.indptr == B.indptr).all():
            raise ValueError("Sparse matrices are not equal: data, indices, or indptr mismatch")
    def pyamg_jacobi(self, max_iter=100, weight=1.0):
        # """PyAMG Jacobi smoother."""
        x_smooth = self.x.copy()    # trickier
        pyamg_smoother.jacobi(self.A, x_smooth, self.b, max_iter, omega=weight)
        return x_smooth
    def pyamg_gauss_seidel(self, max_iter=100):
        """PyAMG Gauss-Seidel smoother."""
        x_smooth = self.x.copy()
        pyamg_smoother.gauss_seidel(self.A, x_smooth, self.b, max_iter)
        return x_smooth
    def pyamg_sor(self, max_iter=100, weight=1.0):
        """PyAMG SOR smoother."""
        x_smooth = self.x.copy()
        pyamg_smoother.sor(self.A, x_smooth, self.b, max_iter, omega=weight)
        return x_smooth
    def simple_jacobi(self, x0, max_iter=100):
        # https://en.wikipedia.org/wiki/Jacobi_method
        D = np.diag(self.A)  # Extract diagonal elements
        # Compared to gauss seidel the diagonal is seperate.
        L_plus_U = self.A - np.diagflat(D)  # Remainder of A (off-diagonal)

        x = x0.copy()
        for _ in range(max_iter):
            x_new = (self.b - np.dot(L_plus_U, x)) / D  # x_new = D^(-1) * (b - (L + U) * x_old)
            x = x_new
            # Can add breaking conditions later.
        return x
    def gauss_seidel(self, x0, max_iter=100):
        # https://en.wikipedia.org/wiki/Gauss%E2%80%93Seidel_method - element based formula
        # Two ways:
        # 1. x_new = L_inv * (b - U.x)
        # 2. x_new = (b - L*x_new - U*x_old)/A_ii
        """Gauss-Seidel smoother."""
        UserWarning("This implementation is still 64 bits, need to change to 32 bits.")
        n = self.A.shape[0]
        x = np.zeros_like(self.b) if x0 is None else x0.copy()

        for k in range(max_iter):
            x_new = np.copy(x)
            
            for i in range(n):
                LX_new = np.dot(self.A[i, :i], x_new[:i])  # Lower triangular
                UX = np.dot(self.A[i, i+1:], x[i+1:])  # Upper triangular
                x_new[i] = (self.b[i] - LX_new - UX) / self.A[i, i]

            x = x_new

        return x

class AMGSolver:
    class eachlevel:
        def __init__(self): 
            # protected variables
            self._level_A = None
            self._level_R = None
            self._level_P = None
            self._level_x = None
            self._level_x_smooth = None
            self._level_b = None
            self._level_residual = None
            ##
        @property
        def level_A(self):
            return self._level_A

        @level_A.setter
        def level_A(self, A):
            self._level_A = A
            
        @property
        def level_R(self):
            return self._level_R
        
        @level_R.setter
        def level_R(self, R):
            self._level_R = R
        
        @property
        def level_P(self):
            return self._level_P
        
        @level_P.setter
        def level_P(self, P):
            self._level_P = P
        
        @property
        def level_x(self):
            return self._level_x
        
        @level_x.setter
        def level_x(self, x):
            self._level_x = x
            
        @property
        def level_x_smooth(self):
            return self._level_x_smooth
        
        @level_x_smooth.setter
        def level_x_smooth(self, x_smooth):
            self._level_x_smooth = x_smooth
            
        @property
        def level_b(self):
            return self._level_b
        
        @level_b.setter
        def level_b(self, b):
            self._level_b = b
        
        @property
        def level_residual(self):
            return self._level_residual
        
        @level_residual.setter
        def level_residual(self, r):
            self._level_residual = r
    
    # coarse_solver = 'cg' or 'direct' or 'pyamg'
    def __init__(self, A_initial, x_initial, b_initial, 
                 maxlevels = 20, 
                 maxiter_smoothing=100, 
                 coarse_solver='cg', data_type=np.float32):
        # protected variables
        self._levels = []    # is a list of eachlevel objects.
        if x_initial.dtype != data_type or b_initial.dtype != data_type or A_initial.dtype != data_type:
            raise TypeError(f"x_initial dtype {x_initial.dtype} is not the same as data_type {data_type}.")
        self._init_x = x_initial.copy()
        self._init_b = b_initial.copy()
        self._init_A = A_initial.copy()
        # other parameters
        self._maxiter_smoothing = maxiter_smoothing
        self._coarse_solver = coarse_solver
        self._maxlevels = maxlevels
        self._dtype = data_type
        
        # setup 
        self.build_levels()

    def coarsen(self, current_level):
        """Simple C/F splitting based on the diagonal dominance."""
        n = current_level.level_A.shape[0]
        coarse_idx = np.arange(0, n, 2)  # Select every second row as coarse
        fine_idx = np.setdiff1d(np.arange(n), coarse_idx)
        return coarse_idx, fine_idx

    def restriction(self, fine_idx, coarse_idx):
        """Simple direct injection restriction."""
        n, nc = len(fine_idx), len(coarse_idx)
        R = sp.lil_matrix((nc, n + nc), dtype=self._dtype)
        for i, ci in enumerate(coarse_idx):
            R[i, ci] = 1  # Direct injection
        return R.tocsr()
    
    def restrict(self, R, r):
        # Restrict the residual
        r_coarse = R @ r
        return r_coarse
    
    def calculate_residual(self, A, b, smooth_x):
        # compute fine grid residual calculation
        r = b - A @ smooth_x
        return r
    
    def interpolation(self, R):
        """Piecewise constant interpolation (transpose of restriction)."""
        return R.T
    
    # setup phase( R, P, A, x)
    def build_levels(self):
        # build levels
        # level 0 initials
        for i in range(self._maxlevels):# allocate all level objects first.
            level = AMGSolver.eachlevel()
            # if it's the first level initialize it.
            if i == 0:
                level.level_A = self._init_A
                level.level_x = self._init_x
                level.level_b = self._init_b
            
            self._levels.append(level)
            
        # R, P, A_next
        for i in range(self._maxlevels-1):  # overflow check
            level = self._levels[i]
            next_level = self._levels[i+1]
            
            # setup steps:
            # 1. m-by-n restrict matrix.(R)
            # 2. n-by-m prolongation matrix.(P)
            # 3. m-by-m coarse matrix.(Galerkin product). Triple sparse matrix            
            coarse_idx, fine_idx = self.coarsen(level)
            level.level_R = self.restriction(fine_idx, coarse_idx)
            level.level_P = self.interpolation(level.level_R)
            next_level.level_A = level.level_R @ level.level_A @ level.level_P

        # x
        for i in range(self._maxlevels):
            if i == 0:
                continue    # As first level values are always input values.
            self._levels[i].level_x = np.zeros(self._levels[i].level_A.shape[0], dtype=self._dtype)
                
        self.print_tabulate_eachlevel("After build_levels(SETUP)")
        
    def solve_V_down(self):

        for i in range(len(self._levels)-1):
            level = self._levels[i]
            next_level = self._levels[i+1]
            
            # Pre-smoothing
            # residual
            # restrict       
            presmoother = smoother(level, level.level_x)
            level.level_x_smooth = presmoother.pyamg_jacobi(max_iter=self._maxiter_smoothing)
            level.level_residual = self.calculate_residual(level.level_A, level.level_b, level.level_x_smooth)
            # next level
            next_level.level_b = self.restrict(level.level_R, level.level_residual)
            
        coarsest_level = self._levels[-1]
        self.print_tabulate_eachlevel("After solve_V_down")
        return coarsest_level.level_b, coarsest_level.level_A
  
    def solve_coarse_solver(self, a_coarse, x_coarse, b_coarse):
        print("Coarse solver details")
        from tabulate import tabulate
        table = [
            ["A_coarse", a_coarse.shape, a_coarse],
            ["x_coarse", x_coarse.shape, "-", x_coarse],
            ["b_coarse", b_coarse.shape, "-", b_coarse]
        ]
        print(tabulate(table, headers=["Variable", "Shape", "NNZ", "values"], tablefmt="simple"))
        
        # Check what solver to use
        if self._coarse_solver == 'cg':
            print("Solver = cg")
            x, _ = scipy_iterative_solver(a_coarse, x_coarse, b_coarse, atol=1.e-12, max_ite=100)
        elif self._coarse_solver == 'direct':
            print("Solver = 'direct'")
            x, _ = scipy_direct_solver(a_coarse, b_coarse)
        else:
            raise ValueError("coarse solver not implemented")
        
        # update last coarse level.
        self._levels[-1].level_x = x    # coarse level
        return x
    
    def print_tabulate_eachlevel(self, print_str="", up = False):
        print(print_str)
        level_info = []
        if up:
            start = len(self._levels)-1
            end = -1
            step = -1
        else:
            start = 0
            end = len(self._levels)
            step = 1
            
        for i in range(start, end, step):
            level = self._levels[i]
            level_info.append([
                i, 
                f"{level.level_A.shape} ({level.level_A.dtype})", 
                level.level_A.nnz, 
                f"{level.level_R.shape} ({level.level_R.dtype})" if level.level_R is not None else "-", 
                f"{level.level_P.shape} ({level.level_P.dtype})" if level.level_P is not None else "-", 
                f"{level.level_x.shape} ({level.level_x.dtype})" if level.level_x is not None else "-", 
                f"{level.level_x_smooth.shape} ({level.level_x_smooth.dtype})" if level.level_x_smooth is not None else "-", 
                f"{level.level_b.shape} ({level.level_b.dtype})" if level.level_b is not None else "-"
            ])
            table = [
                ["Level", i],
                ["A", f"{level.level_A.toarray()} ({level.level_A.dtype})" if level.level_A is not None else '-'],
                ["b", f"{level.level_b} ({level.level_b.dtype})" if level.level_b is not None else '-'],
                ["x", f"{level.level_x} ({level.level_x.dtype})" if level.level_x is not None else '-'],
                ["x_smooth", f"{level.level_x_smooth} ({level.level_x_smooth.dtype})" if level.level_x_smooth is not None else '-'],
                ["R", f"{level.level_R.toarray()} ({level.level_R.dtype})" if level.level_R is not None else '-'],
                ["P", f"{level.level_P.toarray()} ({level.level_P.dtype})" if level.level_P is not None else '-']
            ]
            print(tabulate(table, headers=["Variable", "Values"], tablefmt="simple"))
        print(tabulate(level_info, headers=["LevelID", "Shape of A", "NNZ in A", "Shape of R", "Shape of P", "Shape of x", "Shape of x_smooth", "Shape of b"], tablefmt="simple"))

    def solve_V_up(self, A_coarse, x_coarse, b_coarse):        
        # print the types of the variables

        # 1. prolongation
        # 2. update
        # 3. Post-smoothing
        for i in range(len(self._levels)-2, -1, -1):
            level = self._levels[i]
            prev_level = self._levels[i+1] # may overflow, check later.
            # prolongate and update
            level.level_x_smooth = level.level_x_smooth + level.level_P @ prev_level.level_x
            # post-smoothing
            postsmoother = smoother(level, level.level_x_smooth)
            level.level_x = postsmoother.pyamg_jacobi(max_iter=self._maxiter_smoothing)
        
        self.print_tabulate_eachlevel("After solve_V_up", up=True)
        # calculate rho
        r = self._levels[0].level_b - self._levels[0].level_A.dot(self._levels[0].level_x)
        rho = np.dot(r, r)
        return self._levels[0].level_x, rho   # Return the finest-level solution
