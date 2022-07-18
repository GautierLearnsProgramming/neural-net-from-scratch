import numpy as np


def compute_mse_gradient(X: np.ndarray, y: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    Compute the gradient of the Mean Squared Error with respect to the weights.

    Args:
        X: numpy array of shape (n, d) where d is the number of instances in the batch or mini-batch
        y: numpy array of shape (n, 1)
        weights: numpy array of shape (d + 1, 1)
    Returns:
        numpy array of shape (d + 1, 1) containing the gradient
    """
    aug_X = np.c_[np.ones(X.shape[0]), X]
    return np.transpose(aug_X) @ ((aug_X @ weights).reshape((y.shape[0], y.shape[1])) - y)
