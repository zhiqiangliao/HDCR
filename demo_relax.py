import sys
sys.path.append('../src/')

import numpy as np
import random
import DGP
from src.tools import yhat
from src.constant import convex, concave
from src.estimators import SLasso, CNLS

np.random.seed(0)
random.seed(0)

# set the shape of the function
func = convex

n=100
d=10
s=3
rho=0.3
SNR=3

# generate train and test sample
x, y, y_true, support = DGP.convexfunc_sparse(n+1000, d, s, rho, SNR)

x_tr, y_tr, y_tr_true = x[:n,:], y[:n], y_true[:n]
x_te, y_te, y_te_true = x[-1000:,:], y_true[-1000:], y_true[-1000:]

# fit the CNLS model
alpha, beta = CNLS(x_tr, y_tr, shape=func, positive=False)
# compute the weights for the SLasso model
weights = 1/(np.sum(beta**2, axis=0))
# standardize the weights
weights = weights/np.max(weights)*d

# fit the relaxed SLasso model
lm = 15
gm = 0.1
alpha, beta = SLasso(x_tr, y_tr, w=weights, c=lm, shape=func, positive=False)
support_relax = np.where(np.sum(np.abs(beta), axis=0) > 1e-6)[0]
alpha, beta = SLasso(x_tr[:, support_relax], y_tr, w=weights[support_relax], c=lm*gm, shape=func, positive=False)
y_hat = yhat(alpha, beta, x_te[:, support_relax], shape=func)
risk = np.sum((y_hat - y_te_true)**2)/np.sum(y_te_true**2)

print('Estimated support set of relaxed SLasso:', support_relax)
print('True support set:', support)
print(f"Prediction risk of relaxed SLasso: {risk:.2f}")