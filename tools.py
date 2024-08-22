import numpy as np
from scipy import sparse

from constant import convex, concave
from cvxpy.atoms.norm import norm


# Calculate yhat in testing sample
def yhat(alpha, beta, x_test, shape=convex):
    '''
    function estimate the y_hat of convex functions.
    refers to equation (4.1) in journal article:
    "Representation theorem for convex nonparametric least squares. Timo Kuosmanen (2008)"
    input:
    alpha and beta are regression coefficients; x_test is the input of test sample.
    output:
    return the estimated y_hat.
    '''
    # check the dimension of input
    if beta.shape[1] != x_test.shape[1]:
        raise ValueError('beta and x_test should have the same number of dimensions.')
    else:
        # compute yhat for each testing observation
        yhat = np.zeros((len(x_test),))
        for i in range(len(x_test)):
            if shape == concave:
                yhat[i] = (alpha + np.sum(np.multiply(beta, x_test[i]), axis=1)).min(axis=0)
            elif shape == convex:
                yhat[i] = (alpha + np.sum(np.multiply(beta, x_test[i]), axis=1)).max(axis=0)

    return yhat

def calculate_f1_score(true_support_set, estimated_support_set):
    # Convert sets to lists to ensure compatibility with sklearn
    true_support_list = list(true_support_set)
    estimated_support_list = list(estimated_support_set)
    
    # Calculate true positives, false positives, and false negatives
    true_positives = len(set(true_support_list) & set(estimated_support_list))
    false_positives = len(set(estimated_support_list) - set(true_support_list))
    false_negatives = len(set(true_support_list) - set(estimated_support_list))

    # if there are no true positives, return 0
    if true_positives == 0:
        return 0
    
    # Calculate precision and recall
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    
    # Calculate F1 score
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
    
    return f1   

def _calculate_matrix_A(n):
    '''
    function to calculate matrix A in the constraint A*theta + B*xi >= 0
    '''

    res = np.zeros((n*(n-1), n))
    k = 0
    for i in range(n):
        for j in range(n):
            if i != j:
                res[k, i] = -1
                res[k, j] = 1
                k += 1
    return res

def _calculate_matrix_B(x, n, d):
    '''
    function to calculate matrix B in the constraint A*theta + B*xi >= 0
    '''

    num_rows = n * (n - 1)
    num_cols = n * d

    row_indices = []
    col_indices = []
    data = []

    k = 0
    for i in range(n):
        for j in range(n):
            if i != j:
                row_indices.extend([k] * d)
                col_indices.extend(range(i * d, (i + 1) * d))
                data.extend(x[j, :] - x[i, :])
                k += 1

    sparse_matrix = sparse.coo_matrix((data, (row_indices, col_indices)), shape=(num_rows, num_cols))
    return -sparse_matrix

def _shape_constraint(A, B, Xi, theta, shape=convex, positive=False):
    '''
    function to generate the shape constraint A*theta + B*xi >= 0 or A*theta + B*xi <= 0
    for the optimization problem
    '''

    if shape == convex:
        cons_shape = A @ theta + B @ Xi >= 0
    elif shape == concave:
        cons_shape = A @ theta + B @ Xi <= 0

    if positive:
        cons_positive = Xi >= 0.0
    else:
        return [cons_shape]

    return [cons_shape, cons_positive]

def _Lipschitz_norm(Xi, n, d, l):
    cons_Lipschitz = []
    for i in range(n):
        cons_Lipschitz.append(norm(Xi[i*d:(i+1)*d], 2) <= l)

    return cons_Lipschitz

def _Lipschitz_norm1(Xi, n, d, l):
    cons_Lipschitz = []
    for i in range(n):
        cons_Lipschitz.append(norm(Xi[i*d:(i+1)*d], 1) <= l)

    return cons_Lipschitz

def _L1_norm(Xi, n, d):
    cons_L1 = []
    for i in range(n):
        cons_L1.append(norm(Xi[i*d:(i+1)*d], 1))

    return sum(cons_L1)

def _Linf_norm(Xi, n, d):
    cons_Linf = []
    for i in range(d):
        cons_Linf.append(norm(Xi[i:n*d:d], 'inf'))
    
    return sum(cons_Linf)

def _Linf_weightnorm(Xi, n, d, w):
    cons_Linf = []
    for i in range(d):
        cons_Linf.append(w[i]*norm(Xi[i:n*d:d], 'inf'))
    
    return sum(cons_Linf)


def _bigM_bound(Xi, n, d, bigM, z):
    cons_Up = []
    cons_Low = []
    for i in range(n):
        for j in range(d):
            cons_Up.append(Xi[i*d+j] <= bigM*z[j])
            cons_Low.append(Xi[i*d+j] >= -bigM*z[j])
    cons = cons_Up + cons_Low
    return cons

def _loss_function(y, f, mu, n):
    loss = []
    for i in range(n):
        loss.append((y[i] - sum(f[i,:]) - mu)**2)

    return sum(loss)
