import numpy as np


def compute_mse(x, y):
    """ Compute the Mean Squared Error between vectors x and y

    Args:
        x: numpy array of shape (n, 1)
        y: numpy array of shape (n, 1)

    Returns:
        float: the Mean Squared Error
    """
    return np.transpose(x - y) @ (x - y) / x.shape[0]
