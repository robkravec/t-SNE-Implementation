import numpy as np
np.random.seed(663)
import sys
sys.path.append("../tsne663")
from .tsne import *

###########################################################################################################################
# Get random data to test
###########################################################################################################################
n, p, d = 10, 10, 2
X = np.random.normal(0, 1, [n, p])
R = np.random.gamma(1, 1, [n, n])
Y = np.random.normal(0, 1, [n, d])

###########################################################################################################################
# Test equivalence of results for different levels of optimization
###########################################################################################################################
Y_dists = get_Y_dists(Y)

D_none = get_dists(X, "none")
D_fastest = get_dists(X, "fastest")
assert np.allclose(D_none, D_fastest), "Distance functions return different values"

P_none = get_P(D_fastest, optim = "none")
P_fastest = get_P(D_fastest, optim = "fastest")
assert np.allclose(P_none, P_fastest), "Functions to create p_ij matrices return different values"

grad_none = get_grad(R, Y_dists, Y, "none")
grad_fast = get_grad(R, Y_dists, Y, "fast")
grad_fastest = get_grad(R, Y_dists, Y, "fastest") 
assert np.allclose(grad_none, grad_fast), "Gradient functions return different values"
assert np.allclose(grad_none, grad_fastest), "Gradient functions return different values"

print("Different levels of optimization are consistent.")

###########################################################################################################################
# Check that functions return correct output types
###########################################################################################################################
assert isinstance(D_none, np.ndarray), "Output of distance function with no optimization is not a numpy array"
assert isinstance(D_fastest, np.ndarray), "Output of optimized distance function is not a numpy array" 
assert isinstance(P_none, np.ndarray), "Output of p_ij function with no optimization is not a numpy array" 
assert isinstance(P_fastest, np.ndarray), "Output of optimized p_ij function is not a numpy array"
assert isinstance(grad_none, np.ndarray), "Output of gradient function with no optimization is not a numpy array"
assert isinstance(grad_fast, np.ndarray), "Output of gradient function with some optimization is not a numpy array"
assert isinstance(grad_fastest, np.ndarray), "Output of optimized gradient function is not a numpy array"

tsne_d2 = tsne(X, niter = 3, verbose=False)
assert isinstance(tsne_d2, np.ndarray), "Output of 2D t-SNE function is not a numpy array"
assert tsne_d2.shape[-1] == 2, "Output of t-SNE function with these inputs should be two-dimensional"

tsne_d3 = tsne(X, d=3,  niter = 3, verbose=False)
assert isinstance(tsne_d3, np.ndarray), "Output of 3D t-SNE function is not a numpy array" 
assert tsne_d3.shape[-1] == 3, "Output of t-SNE function with these inputs should be three-dimensional"

print("All functions return the correct output")