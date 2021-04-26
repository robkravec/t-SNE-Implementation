import numpy as np
from numba import jit, njit, prange
from tqdm.notebook import tqdm, trange
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances


###################################################################################
# Utilities.
###################################################################################


def pca(X, k = 30):
    """Use PCA to project X to k dimensions."""
    
    # Center/scale the data.
    s = np.std(X, axis=0)
    s = np.where(s==0, 1, s)
    X = (X - np.mean(X, axis=0))/s
    
    # Run PCA with sklearn.
    pca_ = PCA(n_components=k)
    return pca_.fit_transform(X)


def get_dists(X, optim = "fastest"):
    """Return squared Euclidean pairwise distances."""
    
    if optim == "none":
        n = X.shape[0]
        dists = np.zeros([n, n])
        
        for i in range(n):
            for j in range(i, n):
                dists[i, j] = np.sum((X[i, :] - X[j, :])**2)
 
        return dists + dists.T
    else:
        return pairwise_distances(X, metric="sqeuclidean")
    
    
def entropy_py(p):    
    """Calculates 2 ** H(p) of array p, where H(p) is the Shannon entropy."""
    return 2 ** np.sum(-p*np.log2(p+1e-10))


@jit
def entropy_numba(p):    
    """Calculates 2 ** H(p) of array p, where H(p) is the Shannon entropy."""
    return 2 ** np.sum(-p*np.log2(p+1e-10))
    
    
def get_Y_dists(Y, df = 1, optim = "fastest"):
    """Takes in an n*n matrix Y, returns a matrix of (1+||y_i-y_j||^2/df)^-1."""
    D = get_dists(Y, optim = optim)
    return np.power(1 + D/df, -1)
    

def get_Q(Y_dists):
    """Normalize a matrix excluding the diagonal."""
    np.fill_diagonal(Y_dists, 0)
    return Y_dists/np.sum(Y_dists)


###################################################################################
# Optimizing variances and finding joint probabilities.
###################################################################################


def get_pij(d, scale, i):
    """
    Compute probabilities conditioned on point i from a row of distances
    d and a Gaussian scale (scale = 2*sigma^2). 
    """
    
    d_scaled = -d/scale
    d_scaled -= np.max(d_scaled)
    exp_D = np.exp(d_scaled)
    exp_D[i] = 0
    
    return exp_D/np.sum(exp_D)
    
    
@jit
def get_pij_numba(d, scale, i):
    """
    Compute probabilities conditioned on point i from a row of distances
    d and a Gaussian scale (scale = 2*sigma^2). Uses numba.
    """
    
    d_scaled = -d/scale
    d_scaled -= np.max(d_scaled)
    exp_D = np.exp(d_scaled)
    exp_D[i] = 0
    
    return exp_D/np.sum(exp_D)


def get_P_py(D, target_perp = 30, LB = 0, UB = 1e4, tol = 1e-6, maxit = 250):
    """Optimize standard deviations to target perplexities with binary search. 
    Returns joint probabilities."""
    
    n = D.shape[0]
    P = np.zeros(D.shape)
    
    for i in range(n):
        LB_i = LB
        UB_i = UB
        d = D[i, :]
        
        for t in range(maxit):
            # Find the perplexity using sigma = midpoint.
            midpoint = (LB_i + UB_i)/2
            scale = 2*midpoint**2
            p_ij = get_pij(d, scale, i)
            current_perp = entropy_py(p_ij)

            if current_perp < target_perp:
                LB_i = midpoint
            else:
                UB_i = midpoint

            if np.abs(current_perp-target_perp) < tol:
                break
            
        P[i,:] = p_ij
        
    return (P+P.T)/(2*n)


@njit(parallel = True)
def get_P_numba(D, target_perp = 30, LB = 0, UB = 1e4, tol = 1e-6, maxit = 250):
    """Optimize standard deviations to target perplexities with binary search
    in parallel. Returns joint probabilities."""
    
    n = D.shape[0]
    P = np.zeros(D.shape)
    
    for i in prange(n):
        LB_i = LB
        UB_i = UB
        d = D[i, :]
        
        for t in range(maxit):
            # Find the perplexity using sigma = midpoint.
            midpoint = (LB_i + UB_i)/2
            scale = 2*midpoint**2
            p_ij = get_pij_numba(d, scale, i)
            current_perp = entropy_numba(p_ij)
            
            if current_perp < target_perp:
                LB_i = midpoint
            else:
                UB_i = midpoint

            if np.abs(current_perp-target_perp) < tol:
                break
            
        P[i,:] = p_ij
        
    return (P+P.T)/(2*n)


def get_P(D, target_perp = 30, LB = 0, UB = 1e4, tol = 1e-6, maxit = 250, optim = "fastest"):
    """
    Generates NxN symmetric affinity score matrix from pairwise distances.
    
    Input:
        D -  pairwise distance matrix.
        target_perp - target perplexity of conditional probabilies pj_i
        LB - lower bound in binary search 
        UB - upper bound in binary search
        tol - tolerance in binary search
        maxit - maximum  iterations in binary search
        optim - "none", "fast", or "fastest". Which level of optimization to run.
                
    Output:
        P - NxN symmetric affinity score matrix 
    """
    
    if optim == "none" or optim == "fast":
        return get_P_py(D, target_perp, LB, UB, tol, maxit)
    else:
        return get_P_numba(D, target_perp, LB, UB, tol, maxit)
    

###################################################################################
# Computing the gradient.
###################################################################################

def grad_py(R, Y_dists, Y):
    """Compute the t-SNE gradient with raw Python."""
    
    n = Y.shape[0]
    dY = np.zeros(shape = Y.shape)
    
    for i in range(n):
        for j in range(n):
            dY[i,:] += 4*R[i,j]*(Y[i, :] - Y[j, :])*Y_dists[i, j]

    return dY
    

def grad_numpy(R, Y_dists, Y):
    """Compute the t-SNE gradient with vectorization."""
    dY = np.zeros_like(Y)

    for i in range(Y.shape[0]):
        # Write the sum as a dot product of a vector and a matrix.
        dY[i,:] = 4*np.dot(R[i,:]*Y_dists[i,:], Y[i,:] - Y)
            
    return dY
    

@njit(parallel=True)
def grad_numba(R, Y_dists, Y):
    """Compute the t-SNE gradient in parallel."""
    
    n = Y.shape[0]
    d = Y.shape[1]
    dY = np.zeros(shape = Y.shape)
    
    for i in prange(n):
        for j in range(n):
            for k in prange(d):
                dY[i,k] += 4*R[i,j]*(Y[i, k] - Y[j, k])*Y_dists[i, j]

    return dY



def get_grad(R, Y_dists, Y, optim = "fast"):
    """Compute the t-SNE gradient.
    
    Inputs:
        R - n*n matrix of difference between high/low dimensional affinities.
        Y_dists - n*n matrix of embedded similarities.
        Y - n*d matrix of current embeddings.
        optim - "none", "fast", or "fastest". Which level of optimization to run.

    Outputs:
        dY - n*d matrix of t-SNE gradients."""
    
    if optim == "none":
        return grad_py(R, Y_dists, Y)
    elif optim == "fast":
        return grad_numpy(R, Y_dists, Y)
    else:
        return grad_numba(R, Y_dists, Y)
    
    
###################################################################################
# Learning rates.
###################################################################################


def constant(t, eta_init, last_eta, c = 100):
    """Constant learning rate."""
    return c


def time_based(t, eta_init, last_eta, d = 0.01):
    """Time-based learning rate with decay d."""
    return last_eta/(1+d*t)


def step_based(t, eta_init, last_eta, d = 0.01, r = 50):
    """Step-based learning rate with decay d and rate r."""
    return eta_init*d**np.floor((1+t)/r)


def exponential(t, eta_init, last_eta, d = 0.01):
    """Exponential  learning rate with decay d."""
    return eta_init*np.exp(-d*t)

###################################################################################
# Sanitizing tsne inputs.
###################################################################################

def sanitize_inputs(X, niter, alpha_init, alpha_final, alpha_thr, eta_init,
                    d, exg, exg_thr, perplexity, pca_dims, optim, verbose, df):
    
    """Checks to ensure that all inputs to the tsne function are valid and raises
    and informative assertion error for any inputs that are not valid"""
    
    assert isinstance(X, np.ndarray) and len(X.shape) == 2, "X must be a 2-D numpy array"
    assert (isinstance(niter, int) or isinstance(niter, np.int64)) and niter > 0, "niter must be an integer greater than 0"
    assert isinstance(alpha_init, int) or isinstance(alpha_init, float), "alpha_init must be a number"
    assert isinstance(alpha_final, int) or isinstance(alpha_final, float), "alpha_final must be a number"
    assert alpha_init > 0 and alpha_final > 0, "alpha_init and alpha_final must be greater than 0" 
    assert isinstance(alpha_thr, int) and alpha_thr > 0, "alpha_thr must be an integer greater than 0"
    assert isinstance(eta_init, int) or isinstance(eta_init, float), "eta_init must be a number"
    assert eta_init > 0, "eta_init must be greater than 0"
    assert isinstance(d, int) and d >= 2, "d must be an integer greater than or equal to 2"
    assert isinstance(exg, int) or isinstance(exg, float), "exg must be a number"
    assert exg >= 1, "exg must be greater than or equal to 1"
    assert isinstance(exg_thr, int) and exg_thr >= 1, "exg_thr must be an interger greater than or equal to 1"
    assert isinstance(perplexity, int) or isinstance(perplexity, float), "perplexity must be a number"
    assert perplexity >= 5 and perplexity <= 250, "perplexity must be between 5 and 250"
    assert isinstance(pca_dims, int) and pca_dims >= 2, "pca_dims must be an integer greater than or equal to 2"
    assert optim in ["none", "fast", "fastest"], "optim can only take on values of none, fast, and fastest"
    assert isinstance(verbose, bool), "verbose must be True or False"
    assert isinstance(df, int) or isinstance(df, float), "df must be a number"
    assert df > 0, "df must greater than 0"

###################################################################################
# Main function.
###################################################################################


def tsne(X, niter = 1000, alpha_init = 0.5, alpha_final = 0.8, alpha_thr = 250, 
         eta_init = 100, lr_fun = constant, d = 2, exg = 4, exg_thr = 50, 
         perplexity = 30, pca_dims = 30, optim = "fastest", verbose = True, df = 1):
    """Run t-SNE.
    
    Required inputs: 
        X - NxM matrix
    
    Optional inputs:
        d - dimension of embedding
        perplexity - target perplexity
        niter - number of iterations
        alpha_int - initial value of momentum
        alpha_final - final value of momentum term
        alpha_thr - iteration when momentum changes
        eta_init - initial learning rate
        lr_fun - learning rate function
        exg - multiplicative factor for early exaggeration
        exg_thr - iteration to stop exaggeration
        pca_dims - maximum number of dimensions before preprocessing with PCA
        optim - "none", "fast", or "fastest". Which level of optimization to run
        verbose - bool, whether or not to print a progress bar
        df - degrees of freedom of scaled t-distribution, df=1 is usual t-SNE
    
    Outputs:
        Y - (niter + 2) x N x d array of embeddings for each iteration"""
    
    # Sanitize inputs
    sanitize_inputs(X, niter, alpha_init, alpha_final, alpha_thr, eta_init,
                    d, exg, exg_thr, perplexity, pca_dims, optim, verbose, df)

    # Reduce dimension if needed
    if X.shape[1] > pca_dims:
        X = pca(X, pca_dims)
    
    # Get affinities with exaggeration.
    D = get_dists(X, optim)    
    pij = exg*get_P(D, perplexity, optim=optim)
    
    # Initialize first few iterations.
    size = (pij.shape[0], d)
    Y = np.zeros(shape = (niter + 2, size[0], d))
    initial_vals = np.random.normal(0.0, np.sqrt(1e-4), size)
    Y[0, :, :] = Y_m1 = Y[1, :, :] = Y_m2 = initial_vals
    
    last_eta = eta_init
    alpha = alpha_init
    
    for i in trange(2, niter + 2, disable = not verbose):
        if i == int(alpha_thr):
            # Reduce momentum after some time.
            alpha = alpha_final
        
        if i == int(exg_thr):
            # Stop the exaggeration.
            pij /= exg
        
        # Compute gradient.
        Y_dists = get_Y_dists(Y_m1, df, optim)
        qij = get_Q(Y_dists)
        rij = pij - qij
        grad = get_grad(rij, Y_dists, Y_m1, optim)
        
        # Update learning rate.
        eta = lr_fun(i, eta_init, last_eta)
        last_eta = eta
        
        # Update embeddings.
        Y_new = Y_m1 - eta*grad + alpha*(Y_m1 - Y_m2)
        Y_m2, Y_m1 = Y_m1, Y_new
        Y[i, :, :] = Y_new
    
    return Y