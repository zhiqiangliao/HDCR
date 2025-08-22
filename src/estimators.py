import numpy as np
from cvxpy import Variable, sum_squares, Minimize, Problem

from .tools import _calculate_matrix_A, _calculate_matrix_B, _shape_constraint, _Linf_weightnorm, _L1_norm, _L2_norm, _Lipschitz_norm
from .constant import convex, concave


def SLasso(x, y, w, c=1.0, shape=convex, positive=False):
    n, d = x.shape
    A = _calculate_matrix_A(n)
    B = _calculate_matrix_B(x, n, d)

    # interface with cvxpy
    Xi = Variable(n*d)
    theta = Variable(n)
    objective = 0.5*sum_squares(y - theta)  + c*_Linf_weightnorm(Xi, n, d, w)

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

def Lasso(x, y, c, shape=convex, positive=False):
    n, d = x.shape
    A = _calculate_matrix_A(n)
    B = _calculate_matrix_B(x, n, d)

    # interface with cvxpy
    Xi = Variable(n*d)
    theta = Variable(n)
    objective = 0.5*sum_squares(y - theta)

    # add shape constraint
    constraint = _shape_constraint(A, B, Xi, theta, shape=shape, positive=positive)

    # add Lipschitz constraint
    constraints = constraint + _Lipschitz_norm(Xi, n, d, c)

    # optimize the model with solver
    prob = Problem(Minimize(objective), constraints)
    prob.solve(solver='MOSEK')
    
    Xi_val = Xi.value.reshape(n,d)
    theta_val = theta.value
    alpha = list([theta_val[i] - Xi_val[i,:]@x[i,:] for i in range(n)])

    # # if the element in Xi_val is less than 1e-6, then set it to 0
    # for i in range(n):
    #     for j in range(d):
    #         if abs(Xi_val[i,j]) < 1e-6:
    #             Xi_val[i,j] = 0
    beta = Xi_val

    return alpha, beta

def ENet(x, y, lm, gm, shape=convex, positive=False):
    n, d = x.shape
    A = _calculate_matrix_A(n)
    B = _calculate_matrix_B(x, n, d)

    # interface with cvxpy
    z = Variable(d, boolean=True)
    Xi = Variable(n*d)
    theta = Variable(n)
    objective = 0.5*sum_squares(y - theta) + lm*(gm*_L1_norm(Xi, n, d)+(1-gm)*_L2_norm(Xi, n, d))

    # add shape constraint
    constraint = _shape_constraint(A, B, Xi, theta, shape=shape, positive=positive)

    # optimize the model with solver
    prob = Problem(Minimize(objective), constraint)
    prob.solve(solver='MOSEK')
    
    Xi_val = Xi.value.reshape(n,d)
    theta_val = theta.value
    alpha = list([theta_val[i] - Xi_val[i,:]@x[i,:] for i in range(n)])

    beta = Xi_val

    return alpha, beta

def CNLS(x, y, c=0.1, shape=convex, positive=False):
    n, d = x.shape
    A = _calculate_matrix_A(n)
    B = _calculate_matrix_B(x, n, d)

    Xi = Variable(n*d)
    theta = Variable(n)

    objective = 0.5*sum_squares(y - theta) + c*sum_squares(Xi)

    # add shape constraint
    constraint = _shape_constraint(A, B, Xi, theta, shape=shape, positive=positive)

    # optimize the model with solver
    prob = Problem(Minimize(objective), constraint)
    prob.solve(solver='MOSEK')
    
    Xi_val = Xi.value.reshape(n,d)
    theta_val = theta.value

    alpha = list([theta_val[i] - Xi_val[i,:]@x[i,:] for i in range(n)])
    beta = Xi_val

    return alpha, beta
