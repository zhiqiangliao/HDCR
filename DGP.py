import numpy as np

def convexfunc_sparse(n, d, k, rho, SNR):
    """
    DGP: y = ||x||^2 + e

    Parameters:
    n: int - Number of samples
    d: int - Number of dimensions/features
    k: int - Number of non-zero dimensions in the support set
    SNR: float - Signal-to-noise ratio
    rho: float - Correlation coefficient between variables, 0 < rho < 1
    """
    
    # Generate the correlation matrix Sigma = rho^|i-j|
    indices = np.arange(d)
    Sigma = rho ** np.abs(np.subtract.outer(indices, indices))
    
    # Generate data from multivariate normal distribution
    x = np.random.multivariate_normal(mean=np.zeros(d), cov=Sigma, size=n)
    
    # Generate the support set with equal spacing
    # Ensure k is less than d to avoid index error
    if k >= d:
        raise ValueError("k must be less than d")
    support = np.round(np.linspace(0, d - 1, k)).astype(int)
    x_support = x[:, support]

    # Compute true output
    y_true = np.linalg.norm(x_support, axis=1)**2
    
    # Calculate noise standard deviation
    sigma = np.sqrt(np.var(y_true, ddof=1) / SNR)
    
    # Generate noise
    nse = np.random.normal(0, sigma, n)
    
    # Compute observed output
    y = y_true + nse
    
    return x, y, y_true, support