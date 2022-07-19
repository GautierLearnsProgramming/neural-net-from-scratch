import numpy as np


def log_loss(x, y):
    return np.mean(- y * np.log(np.maximum(x, 1e-6) - (1 - y) * np.log(np.maximum(1 - x, np.array(1e-6)))))


def log_loss_gradient(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """ Compute the gradient of the log loss cost function.

    Args:
        x: the predicted vector (ndarray of shape (batch_size, n))
        y: the true vector (ndarray of shape (batch_size, n))
    Returns:
        The gradient of the log loss cost function with respect to the predicted vector
        (ndarray of shape (batch_size, n))
    """
    return x - y
