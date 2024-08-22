import numpy as np
from cvxpy import Variable, sum_squares, Minimize, Problem

from tools import _calculate_matrix_A, _calculate_matrix_B, _shape_constraint, _Linf_norm
from constant import convex, concave


def SLasso(x, y, c=1.0, shape=convex, positive=False):
    n, d = x.shape
    A = _calculate_matrix_A(n)
    B = _calculate_matrix_B(x, n, d)

    # interface with cvxpy
    Xi = Variable(n*d)
    theta = Variable(n)
    objective = 0.5*sum_squares(y - theta)  + c*_Linf_norm(Xi, n, d)

    # add shape constraint
    constraint = _shape_constraint(A, B, Xi, theta, shape=shape, positive=positive)

    # optimize the model with solver
    prob = Problem(Minimize(objective), constraint)
    prob.solve(solver='MOSEK')
    
    Xi_val = Xi.value.reshape(n,d)
    theta_val = theta.value

    alpha = list([theta_val[i] - Xi_val[i,:]@x[i,:] for i in range(n)])
    
    # if each element in a column of Xi_val is less than 1e-6, then set it to 0
    for i in range(d):
        if np.max(abs(Xi_val[:,i])) < 1e-6:
            Xi_val[:,i] = np.zeros(n)
    beta = Xi_val

    return alpha, beta