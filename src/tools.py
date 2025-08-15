import numpy as np
from scipy import sparse

from .constant import convex, concave
from cvxpy.atoms.norm import norm
from cvxpy import reshape


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

def fyhat(f, beta, mu, x_train, x_test, fun=convex):

    n, d = x_train.shape
    n_test, d_test = x_test.shape

    yhat = np.zeros((n_test, d_test))

    for j in range(n_test):
        for k in range(d):
            for i in range(n):
                if fun == concave:
                    yhat[j,k] = (f[i,k] + beta[i,k]*(x_test[j,k] - x_train[i,k])).min(axis=0)
                elif fun == convex:
                    yhat[j,k] = (f[i,k] + beta[i,k]*(x_test[j,k] - x_train[i,k])).max(axis=0)

    fyhat = np.sum(yhat, axis=1) + mu
    
    return fyhat

# calculate the index of training set
def index_tr(k, i_kfold):
    
    i_kfold_without_k = i_kfold[:k] + i_kfold[(k + 1):]
    flatlist = [item for elem in i_kfold_without_k for item in elem]

    return flatlist

def calculate_jaccard_similarity(selected_set):
    """
    Calculate the Jaccard similarity for a set of selected sets.
    
    Parameters:
    selected_set: list of sets - Simulated selected variable sets from different runs
    
    Returns:
    float - The Jaccard similarity score.
    """
    if not selected_set:
        return 0.0
    
    # Compute Jaccard similarities for all pairs
    jaccard_similarities = []
    for i in range(len(selected_set)):
        for j in range(i + 1, len(selected_set)):
            intersection = np.intersect1d(selected_set[i], selected_set[j])
            union = np.union1d(selected_set[i], selected_set[j])
            if len(union) > 0:
                jaccard_similarity = len(intersection) / len(union)
            else:
                jaccard_similarity = 0.0
            jaccard_similarities.append(jaccard_similarity)

    # Return the average Jaccard similarity
    if jaccard_similarities:
        return np.mean(jaccard_similarities)
    else:
        return 0.0

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

def _L1_norm(Xi, n, d):
    cons_L1 = []
    for i in range(n):
        cons_L1.append(norm(Xi[i*d:(i+1)*d], 1))

    return sum(cons_L1)

def _L2_norm(Xi, n, d):
    cons_L2 = []
    for i in range(n):
        cons_L2.append(norm(Xi[i*d:(i+1)*d], 2))

    return sum(cons_L2)

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

def _fshape_constraint(n, d, beta, shape=convex, positive=False):
    cons_shape = []

    for k in range(d):
        for i in range(n-2):
            if shape == convex:
                cons_shape.append(beta[i+1,k] - beta[i,k] >= 0)
            elif shape == concave:
                cons_shape.append(beta[i+1,k] - beta[i,k] <= 0)

    if positive:
        cons_positive = beta >= 0.0
        return cons_shape.append(cons_positive)
    else:
        return cons_shape

def _zero_constraint(f, d):
    cons_zero = []
    for k in range(d):
        cons_zero.append(sum(f[:,k]) == 0)

    return cons_zero

def _regression_constraint(f, x, beta, n, d):
    cons_reg = []
    for k in range(d):
        for i in range(n-1):
            cons_reg.append(f[i+1,k] == f[i,k] + beta[i,k]*(x[i+1,k] - x[i,k]))

    return cons_reg

def _finf_norm(f, d):
    cons_inf = []
    for i in range(d):
        cons_inf.append(norm(f[:,i], 'inf'))

    return sum(cons_inf)

def _fl2_norm(f, d):
    cons_l2 = []
    for i in range(d):
        cons_l2.append(norm(f[:,i], 1))

    return sum(cons_l2)

def variable_selection(beta):
    n, d = beta.shape
    idx = [i for i in range(d) if np.max(beta[:,i]) != 0]

    return idx

def trans_list(li):
    if type(li) == list:
        return li
    return li.tolist()


def to_1d_list(li):
    if type(li) == int or type(li) == float:
        return [li]
    if type(li[0]) == list:
        rl = []
        for i in range(len(li)):
            rl.append(li[i][0])
        return rl
    return li


def to_2d_list(li):
    if type(li[0]) != list:
        rl = []
        for value in li:
            rl.append([value])
        return rl
    return li

def x_sort(x):
    x = np.asarray(x)
    x = np.sort(x, axis=0)
    return x