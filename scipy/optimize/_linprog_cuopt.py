"""cuOpt Linear Optimization Methods

Interface to cuOpt linear optimization software.
https://docs.nvidia.com/cuopt

.. versionadded:: 25.02

References
----------

"""
from cuopt.linear_programming.data_model import DataModel
from cuopt.linear_programming.solver import Solve
from cuopt.linear_programming.solver_settings import (
    PDLPSolverMode,
    SolverMethod,
    SolverSettings,
)
from cuopt.linear_programming.solver.solver_parameters import (
    CUOPT_PDLP_SOLVER_MODE,
    CUOPT_METHOD,
    CUOPT_LOG_TO_CONSOLE
)
    
import numpy as np
from scipy import sparse
   
from ._optimize import OptimizeResult, OptimizeWarning

import warnings

cuopt_to_scipy =  {
    0: (4, "cuopt no termination code"),
    1: (0, ""),
    2: (4, "cuopt numerical error"),
    3: (2, "cuopt primal infeasible"),
    4: (3, "cuopt dual infeasible"),
    5: (1, "cuopt iteration limit"),
    6: (1, "cuopt time limit"),
    7: (1, "cuopt primal feasible")
} 


def solver_mode(m):
    try:
        if m.isdigit():
            return PDLPSolverMode(int(m))
        return PDLPSolverMode[m]
    except Exception:
        message = f"bad value for cuOpt pdlp_solver_mode: {m}"
        warnings.warn(message, RuntimeWarning)        
        return None

def solver_method(m):
    try:
        if m.isdigit():
            return SolverMethod(int(m))
        return SolverMethod[m]
    except Exception:
        message = f"bad value for cuOpt method: {m}"
        warnings.warn(message, RuntimeWarning)        
        return None

def _get_cuopt_parameter_strings():
    from cuopt.linear_programming.solver import solver_parameters
    
    # Get all attributes that start with CUOPT_
    cuopt_attrs = [attr for attr in dir(solver_parameters) if attr.startswith('CUOPT_')]
    
    # Extract string values
    result = []
    for attr in cuopt_attrs:
        value = getattr(solver_parameters, attr)
        if isinstance(value, str):
            result.append(value)
    
    return result

def _get_solver_settings(solver_opts, mip):
    ss = SolverSettings()
    if solver_opts is None:
        return ss
        
    # rename maxiter to iteration_limit
    if "maxiter" in solver_opts:
        solver_opts["iteration_limit"] = solver_opts["maxiter"]
        solver_ops.pop("maxiter")
            
    # "disp" takes precedence only if True, otherwise check CUOPT_LOG_TO_CONSOLE
    v = solver_opts.pop("disp", False) or solver_opts.get(CUOPT_LOG_TO_CONSOLE, False)
    solver_opts[CUOPT_LOG_TO_CONSOLE] = v

    if CUOPT_PDLP_SOLVER_MODE in solver_opts:
        m = solver_mode(solver_opts[CUOPT_PDLP_SOLVER_MODE])
        if m is not None:
            ss.set_parameter(CUOPT_PDLP_SOLVER_MODE, m)
        solver_opts.pop(CUOPT_PDLP_SOLVER_MODE)

    if CUOPT_METHOD in solver_opts:
        m =  solver_method(solver_opts[CUOPT_METHOD])
        if m is not None:
            ss.set_parameter(CUOPT_METHOD, m)
        solver_opts.pop(CUOPT_METHOD)

    if "optimality" in solver_opts:
        ss.set_optimality_tolerance(solver_opts["optimality"])
        solver_opts.pop("optimality")

    valid = _get_cuopt_parameter_strings()
    for p,v in solver_opts.items():
        if p in valid:
            ss.set_parameter(p, solver_opts[p])
        else:
            message = f"cuOpt unrecognized parameter {p}"
            warnings.warn(message, RuntimeWarning, stacklevel=3)

    return ss


def _combine_constraints2(A_ub, b_ub, A_eq, b_eq):
    """Combine inequality and equality constraints into a single CSR matrix."""
    constraint_matrices = []
    lower_bounds = []
    upper_bounds = []

    if A_ub is not None:
        # Convert to CSR if sparse, otherwise convert to numpy array
        if sparse.issparse(A_ub):
            A_ub = A_ub.tocsr()
            n_rows = A_ub.shape[0]
        else:
            A_ub = np.array(A_ub, dtype=np.float64)
            n_rows = len(A_ub)
        
        constraint_matrices.append(A_ub)
        lower_bounds.extend([-np.inf] * n_rows)  # -inf for inequalities
        upper_bounds.extend(b_ub)                # upper bound from b_ub

    if A_eq is not None:
        # Convert to CSR if sparse, otherwise convert to numpy array
        if sparse.issparse(A_eq):
            A_eq = A_eq.tocsr()
            n_rows = A_eq.shape[0]
        else:
            A_eq = np.array(A_eq, dtype=np.float64)
            n_rows = len(A_eq)
            
        constraint_matrices.append(A_eq)
        lower_bounds.extend(b_eq)  # equality constraints have same upper
        upper_bounds.extend(b_eq)  # and lower bounds

    if constraint_matrices:
        if all(sparse.issparse(m) for m in constraint_matrices):
            # If all matrices are sparse, use sparse vstack
            combined = sparse.vstack(constraint_matrices)
        else:
            # If any matrix is dense, convert all to dense and use np.vstack
            dense_matrices = [m.toarray() if sparse.issparse(m) else m 
                            for m in constraint_matrices]
            combined = np.vstack(dense_matrices)
            combined = sparse.csr_matrix(combined)
        
        return combined, np.array(lower_bounds), np.array(upper_bounds)
    else:
        return None, np.array([]), np.array([])

def _combine_constraints(A_ub, b_ub, A_eq, b_eq):
    """Combine inequality and equality constraints into a single CSR matrix."""
    constraint_matrices = []
    constraint_types = []
    constraint_bounds = []

    if A_ub is not None:
        # Convert to CSR if sparse, otherwise convert to numpy array
        if sparse.issparse(A_ub):
            A_ub = A_ub.tocsr()
            n_rows = A_ub.shape[0]
        else:
            A_ub = np.array(A_ub, dtype=np.float64)
            n_rows = len(A_ub)
        
        constraint_matrices.append(A_ub)
        constraint_types.extend(['L'] * n_rows)
        constraint_bounds.extend(b_ub)

    if A_eq is not None:
        # Convert to CSR if sparse, otherwise convert to numpy array
        if sparse.issparse(A_eq):
            A_eq = A_eq.tocsr()
            n_rows = A_eq.shape[0]
        else:
            A_eq = np.array(A_eq, dtype=np.float64)
            n_rows = len(A_eq)
            
        constraint_matrices.append(A_eq)
        constraint_types.extend(['E'] * n_rows)
        constraint_bounds.extend(b_eq)

    if constraint_matrices:
        if all(sparse.issparse(m) for m in constraint_matrices):
            # If all matrices are sparse, use sparse vstack
            combined = sparse.vstack(constraint_matrices)
        else:
            # If any matrix is dense, convert all to dense and use np.vstack
            dense_matrices = [m.toarray() if sparse.issparse(m) else m 
                            for m in constraint_matrices]
            combined = np.vstack(dense_matrices)
            combined = sparse.csr_matrix(combined)
        
        return combined, constraint_types, constraint_bounds
    else:
        return None, [], []

def trim_zero_last_column(csr):
    """
    Check if last column is all zeros and remove it if so.
    Uses dense matrix representation for checking and trimming.
    Returns: trimmed CSR if last column was zeros, original CSR if not
    """
    # Convert to dense
    dense = csr.toarray()
    
    # Check if last column is all zeros
    last_col = dense[:, -1]
    if np.allclose(last_col, 0, rtol=1e-10, atol=1e-10):
        print("Last column is all zeros - trimming it")
        # Remove last column and convert back to CSR
        trimmed_dense = dense[:, :-1]
        return sparse.csr_matrix(trimmed_dense)
    else:
        print(f"Last column has non-zero elements - keeping it")
        nonzero_rows = np.nonzero(last_col)[0]
        print(f"Non-zero elements in rows: {nonzero_rows}")
        print(f"Values: {last_col[nonzero_rows]}")
        return csr

    
def _linprog_cuopt(lp, options):
    """
    """
    c, A_ub, b_ub, A_eq, b_eq, bounds, x0, integrality = lp
    
    #csr_matrix, constraint_types, constraint_bounds = _combine_constraints(A_ub, b_ub, A_eq, b_eq)
    csr_matrix, lower_bounds, upper_bounds = _combine_constraints2(A_ub, b_ub, A_eq, b_eq)

    csr_matrix = trim_zero_last_column(csr_matrix)
    
    #if A_ub is not None:
    #    constraint_matrices.append(np.array(A_ub, dtype=np.float64))
    #    constraint_types.extend(['L'] * len(A_ub))
    #    constraint_bounds.extend(b_ub)

    #if A_eq is not None:
    #    constraint_matrices.append(np.array(A_eq, dtype=np.float64))
    #    constraint_types.extend(['E'] * len(A_eq))
    #    constraint_bounds.extend(b_eq)

    #if constraint_matrices:
    #    combined = np.vstack(constraint_matrices)
    #    csr_matrix = sparse.csr_matrix(combined)

    #else:
    #    csr_matrix = None

    import datetime
    print(f"in linprog_cuopt {datetime.datetime.now()}")
    
    #constraint_types = np.array(constraint_types, dtype='U1') if constraint_types else None
    #constraint_bounds = np.array(constraint_bounds, dtype=np.float64) if constraint_bounds else None

    # Check if it's a 2-element tuple
    if len(bounds) == 2 and all(x is None or isinstance(x, (int, float)) for x in bounds):
        var_lower_bounds = np.array([bounds[0]] * len(c))
        var_upper_bounds = np.array([bounds[1]] * len(c))
    else:
        var_lower_bounds = np.array([b[0] if b[0] is not None else -np.inf for b in bounds], dtype=np.float64)
        var_upper_bounds = np.array([b[1] if b[1] is not None else np.inf for b in bounds], dtype=np.float64)

    if integrality is not None:
        var_types = np.array(["I" if x in [1, 3] else "C" for x in integrality], dtype="U1")
    else:
        var_types = np.empty(len(c), dtype='U1')
        var_types.fill('C')
    coeffs = np.array(c, dtype=np.float64)
    
    data_model = DataModel()
    data_model.set_csr_constraint_matrix(csr_matrix.data, csr_matrix.indices, csr_matrix.indptr)
    data_model.set_constraint_lower_bounds(lower_bounds)
    data_model.set_constraint_upper_bounds(upper_bounds)
    #data_model.set_constraint_bounds(constraint_bounds)
    #data_model.set_row_types(constraint_types)
    
    data_model.set_variable_lower_bounds(var_lower_bounds)
    data_model.set_variable_upper_bounds(var_upper_bounds)
    data_model.set_variable_types(var_types)
    data_model.set_objective_coefficients(coeffs)

    ss = _get_solver_settings(options, mip = (integrality is not None and (1 in integrality or 3 in integrality)))

    import pickle
    if True:
        with open("cuopt.pickle", "wb") as f:
            pickle.dump(ss, f)
            pickle.dump(csr_matrix.data, f)
            pickle.dump(csr_matrix.indices, f)
            pickle.dump(csr_matrix.indptr, f)
            pickle.dump(coeffs, f)
            pickle.dump(lower_bounds, f)
            pickle.dump(upper_bounds, f)
            pickle.dump(var_lower_bounds, f)
            pickle.dump(var_upper_bounds, f)
            pickle.dump(var_types, f)
    
    print(f"in calling solve {datetime.datetime.now()}")    
    res = Solve(data_model, ss)
    print(f"back from solve {datetime.datetime.now()}")        
    primal_solution = res.get_primal_solution()

    # Calculate slack variables for inequalities
    slack = None
    if A_ub is not None and b_ub is not None:
        slack = b_ub - A_ub @ primal_solution

    # Calculate residuals for equalities
    con = None
    if A_eq is not None and b_eq is not None:
        con = b_eq - A_eq @ primal_solution
        
    status, message = cuopt_to_scipy[res.get_termination_status()]

    sol = {"status": status,
           "message": message,
           "x": primal_solution,
           "fun": res.get_primal_objective(),
           "slack": slack,
           "con": con,
           "nit": None
    }

    if res.get_problem_category() == 0:
        sol["nit"] = res.get_lp_stats()["nb_iterations"]

    return sol
    
