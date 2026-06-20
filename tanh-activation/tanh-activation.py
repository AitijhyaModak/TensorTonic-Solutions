import numpy as np

def tanh(x):
    """
    Implement Tanh activation function.
    """
    x = np.array(x)
    e_pos = np.exp(x)
    e_neg = np.exp(-x)

    return (e_pos - e_neg) / (e_pos + e_neg)
    pass