def elu(x, alpha):
    """
    Apply ELU activation to each element.
    """
    import numpy as np
    x = np.array(x)

    mul = np.exp(x) - 1

    return np.where(x > 0, x, alpha * mul).tolist()
    