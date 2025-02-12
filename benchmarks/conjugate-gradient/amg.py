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


# solve a linear system A * x = b
# where A is a symmetric positive definite matrix
# The AMG algorithm is from pyamg, https://github.com/pyamg/pyamg

# only V cycle.
def AMG_only_solve(A_csr, b0, x0, tol, max_ite, max_levels, max_coarse, solver='cg'):
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
    ml, setup_config = smooth_aggregate_setup_only(A_csr, x0, b0, max_ite, tol, solver_callable=solver)
    # print_table_data(ml.levels)
    
    level_data_clone = []

    x_level = [x0, ]
    # if only 1 level solve directly
    if len(ml.levels) == 1:
        # x, rho = scipy_iterative_solver(A_csr, x0, b0, tol, max_ite=max_ite)
        x_coarsest[:] = solver(A_csr, b0)
        return x, rho

    # solve phase layer by layer.
    for i, level in enumerate(ml.levels[:-1]):
        A, b, P, R = level.A, level.B, level.P, level.R
        x = x_level[i]
        b_coarse = ml.levels[i+1].B
        
        level.presmoother(A, x, b)   
        r = b - A @ x.reshape(-1, 1)
        b_coarse = (R @ r)    # b_coarse
        x_level.append(np.zeros((b_coarse.shape[0],), dtype=np.float32))
        
        # fill table_data_clone using tabulate
        level_data_clone.append({
            "level": i,
            "A_shape": A.toarray()[0,0:5],
            "B_shape": b[0:5],
            "P_shape": P.shape,
            "R_shape": R.shape,
            "b_coarse_shape": b_coarse.shape,
            "x_coarse_shape": x_level[-1][0:5]
        })

    # solve coarse
    b_coarsest = ml.levels[-1].B
    x_coarsest = x_level[-1]
    A_coarsest = ml.levels[-1].A
    # x_coarsest[:], rho = scipy_iterative_solver(A_coarsest, x_coarsest, b_coarsest, tol, max_ite)
    x_coarsest[:] = solver(A_coarsest, b_coarsest)
    
    level_data_clone.append({
        "level": len(ml.levels)-1,
        "A_shape": A_coarsest.toarray()[0, 0:5],
        "b_coarse_shape": b_coarsest[0:5],
        "x_coarse_shape": x_coarsest[0:5]
    })
    
    # solve up
    for i in range(len(ml.levels)-2, -1, -1):
        prev_level = ml.levels[i+1]
        level = ml.levels[i]
        
        A, b, P = level.A, level.B, level.P
        x = x_level[i]
        prev_x = x_level[i+1]
                    
        x = x + P @ prev_x
        level.postsmoother(A, x, b)
        
        
    print(tabulate(level_data_clone, headers="keys", tablefmt="grid"))
    # rho
    r = ml.levels[0].B - ml.levels[0].A @ x_level[0]
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

        
#############################################
def pyamg(A, x, b, relative_tol, max_ite, cycle_type='V', solver='cg'):
    import pyamg
    from scipy.sparse import random
    # Use PyAMG for solving the system
    ml = pyamg.ruge_stuben_solver(A, coarse_solver=solver)  # Multi-level solver

    # Solve the system
    # Use relative tol as this solver expects it, read documentation.
    x_pyamg, info = ml.solve(b, tol=relative_tol, cycle=cycle_type, maxiter=max_ite, return_info=True)
    if info != 0:
        print("PyAMG did not converge, maybe try changing it's parameters. halted at iterations: ", info)

    r = b - A.dot(x_pyamg)
    rho = np.dot(r, r)
    return x_pyamg, rho

def scipy_direct_solver(A, b):
    # print 5 elements from A anf b
    A_str = np.array2string(A.toarray()[0, 0:5], separator=', ')
    B_str = np.array2string(b[0:5].flatten(), separator=', ')
    print(f"A_COARSE[0:5]: {A_str}")
    print(f"B_COARSE[0:5]: {B_str}")
    x = spla.spsolve(A, b)
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
def test_rs_baseline(A, x, b, max_ite, absolute_tol, solver_callable='cg'):

    #   coarse_solver_callable = 'cg'
    #   coarse_solver_callable = test_direct_solver_scipy   # foo(A,b) -> x
    #   coarse_solver_callable = (test_direct_solver_scipy, {"x": x})
    
      coarse_solver_callable = solver_callable

      # 2. config
      ruge_stuben_config = {
        'strength': ('classical', {'theta': 0.01}),  # Gradual coarsening
        'CF': ('RS', {'second_pass': True}),  # Standard RS coarsening
        'interpolation': 'direct',  # Slowest interpolation method
        'presmoother': ('gauss_seidel', {'sweep': 'symmetric'}),  
        'postsmoother': ('gauss_seidel', {'sweep': 'symmetric'}),  
        'max_levels': 50,  # Allow many levels
        'max_coarse': 16,  # Ensure last level has exactly 16 unknowns
        'keep': False,  
        'coarse_solver': coarse_solver_callable  
      }

      # 3. solver
    #   ml = ruge_stuben_solver(A, **ruge_stuben_config)
      ml = ruge_stuben_solver(A)
      
      print(ml)
      # 4. solve
      residual = []
      solve_config = {
            'maxiter': max_ite,
            'cycle': 'V',
            'residuals': residual,
            'tol': absolute_tol,
            'accel': None,
            'callback': None,
            'cycles_per_level': 1,
            'return_info': False,
      }
      x_sol = ml.solve(b, x0=x, **solve_config)
      # 5. assert
      r = b - A*x_sol
      rho = np.dot(r, r)
      return x_sol, rho

def smooth_aggregate_baseline(A, x, b, max_ite, relative_tol, solver_callable='cg'):
    
    #   coarse_solver_callable = test_direct_solver_scipy   # foo(A,b) -> x
    #   coarse_solver_callable = (test_direct_solver_scipy, {"x": x})
    ml, setup_config = smooth_aggregate_setup_only(A, x, b, max_ite, relative_tol, solver_callable=solver_callable)
    # print_table_data(ml.levels)    
    
    # # 4. solve
    residual = []
    x_vals = []

    solve_config = {
        'maxiter': 1,
        'cycle': 'V',
        'residuals': residual,
        'tol': relative_tol,
        'accel': None,
        'callback': None,
        'cycles_per_level': 1,
        'return_info': False,
    }
    # create a callback method which prints the rho and tol at each iteration
    x_sol = ml.solve(b, x0=x, **solve_config)

    r = b - A*x_sol
    rho = np.dot(r, r)
    print_table_data(ml.levels)
    return x_sol, rho

def smooth_aggregate_setup_only(A, x, b, max_ite, relative_tol, solver_callable='cg'):
    
    #   coarse_solver_callable = test_direct_solver_scipy   # foo(A,b) -> x
    #   coarse_solver_callable = (test_direct_solver_scipy, {"x": x})
    
    coarse_solver_callable = solver_callable
    # coarse_solver_callable = 'cg'
    # 2. config

    smoothed_aggregation_solver_config = {
        'B': b,
        'BH': None,
        'symmetry': 'symmetric',
        'aggregate': ('lloyd', {'ratio': 0.70}),  # Reduce coarsening aggressiveness
        'strength': ('symmetric', {'theta': 0.05}),  # Capture more connections
        'smooth': 'jacobi',
        'presmoother': ('jacobi', {'omega': 1.0/3.0, 'iterations': 5}),
        'postsmoother': ('jacobi', {'omega': 1.0/3.0, 'iterations': 5}),
        'improve_candidates': (('gauss_seidel', {'sweep': 'symmetric', 'iterations': 6}), None),
        'max_levels': 20,  
        'max_coarse': 27,  # Keep more unknowns at the coarsest level
        'diagonal_dominance': False,
        'keep': False,
        'coarse_solver': coarse_solver_callable
    }

    # 3. solver
    np.random.seed(100)  
    # This is essential to pass seed as smooth is non-deterministic, 
    # https://github.com/pyamg/pyamg/issues/350
    ml = smoothed_aggregation_solver(A, **smoothed_aggregation_solver_config)

    # print_table_data(ml.levels)
    # print(ml)
    # print_table_shapes(ml.levels)

    setup_config = smoothed_aggregation_solver_config
    
    return ml, setup_config

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
