import numpy as np

def softmax(x):
    """
    Compute the softmax of input x.
    Works for 1D or 2D NumPy arrays.
    For 2D, compute row-wise softmax.
    """

    x = np.array(x)
    max = np.max(x, axis=-1, keepdims=True)
    
    sum = np.sum(np.exp(x - max), axis=-1, keepdims=True)
    x = x - np.max(x, axis=-1, keepdims=True)
    print(sum)
    return np.exp(x) / sum
    