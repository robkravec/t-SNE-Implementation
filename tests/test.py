import numpy as np
from tsne663 import *
np.random.seed(663)

###################################################################################
# Get random data to test
###################################################################################
n, p, d = 10, 10, 2
X = np.random.normal(0, 1, [n, p])
R = np.random.gamma(1, 1, [n, n])
Y = np.random.normal(0, 1, [n, d])

###################################################################################
# Test equivalence of results for different levels of optimization
###################################################################################
Y_dists = get_Y_dists(Y)

D_none = get_dists(X, "none")
D_fastest = get_dists(X, "fastest")
assert np.allclose(D_none, D_fastest)

P_none = get_P(D_fastest, optim = "none")
P_fastest = get_P(D_fastest, optim = "fastest")
assert np.allclose(P_none, P_fastest)

grad_none = get_grad(R, Y_dists, Y, "none")
grad_fast = get_grad(R, Y_dists, Y, "fast")
grad_fastest = get_grad(R, Y_dists, Y, "fastest")

assert np.allclose(grad_none, grad_fast)

assert np.allclose(grad_none, grad_fastest)

print("Different levels of optimization are consistent.")

###################################################################################
# Check that functions return correct output types
################################################################################### 
assert isinstance(D_none, np.ndarray) 
assert isinstance(D_fastest, np.ndarray) 
assert isinstance(P_none, np.ndarray) 
assert isinstance(P_fastest, np.ndarray)
assert isinstance(grad_none, np.ndarray)
assert isinstance(grad_fast, np.ndarray)
assert isinstance(grad_fastest, np.ndarray)

tsne_d2 = tsne(X, niter = 3, verbose=False)
assert isinstance(tsne_d2, np.ndarray) and tsne_d2.shape[-1] == 2

tsne_d3 = tsne(X, d=3,  niter = 3, verbose=False)
assert isinstance(tsne_d3, np.ndarray) and tsne_d3.shape[-1] == 3

print("All functions return the correct output")