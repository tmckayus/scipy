import numpy as np
from scipy import sparse
from cuopt.linear_programming.data_model import DataModel
from cuopt.linear_programming.solver import Solve
from cuopt.linear_programming.solver_settings import SolverSettings
from ._linprog_cuopt import _get_solver_settings

cuopt_to_scipy =  {
    0: (4, "cuopt no termination code"),
    1: (0, ""),
    2: (4, "cuopt feasible found"),
    3: (2, ""),
    4: (3, ""),
    5: (1, "cuopt iteration limit"),
}
        
def _cuopt_milp(c, indptr, indices, data, b_l, b_u,
                lb, ub, integrality, options):
    data_model = DataModel()
    data_model.set_csr_constraint_matrix(data, indices, indptr)        
    data_model.set_constraint_lower_bounds(b_l)
    data_model.set_constraint_upper_bounds(b_u)
    data_model.set_variable_types(np.array(["I" if x in [1, 3] else "C" for x in integrality], dtype="U1"))
    data_model.set_variable_lower_bounds(lb)
    data_model.set_variable_upper_bounds(ub)
    data_model.set_objective_coefficients(c)

    ss = _get_solver_settings(options, mip=1 in integrality or 3 in integrality)

    return Solve(data_model, ss)
