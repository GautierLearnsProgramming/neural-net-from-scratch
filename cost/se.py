import numpy as np


def squared_error(z: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.sum((z - y) ** 2)


def compute_se_gradient(z: np.ndarray, y: np.ndarray) -> np.ndarray:
    """ Compute the gradient of the squared error cost function.

    Args:
        z: the predicted vector (ndarray of shape (batch_size, n))
        y: the true vector (ndarray of shape (batch_size, n))
    Returns:
        The gradient of the squared error cost function with respect to the predicted vector
        (ndarray of shape (batch_size, n))
    """
    return 2 * (z - y)
