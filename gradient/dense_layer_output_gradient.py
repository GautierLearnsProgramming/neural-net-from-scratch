import numpy as np
from typing import Callable


def compute_layer_output_gradient(y: np.ndarray, dJdz: np.ndarray, w: np.ndarray,
                                  activation_function_dif: Callable[[float], float]) -> np.ndarray:
    """ Compute the gradient of the loss function with respect to the output of the layer.
    This function compute the gradient relative to the output of the layer k-1

    Args:
        y: The output of the k-1 layer (ndarray of shape (1, n))
        dJdz: The gradient of the loss function with respect to the output of the k layer (ndarray of shape (1, m))
        w: The weights of the k layer (ndarray of shape (n, m))
        activation_function_dif: The derivative of the activation function of the k layer.
    Returns:
        The gradient of the loss function with respect to the output of the k layer (ndarray of shape (1, n))
    """
    h = np.vectorize(activation_function_dif)
    dzdy = w * h(y @ w)  # (n, m)
    dJdy = dJdz @ dzdy.T  # (1, n)
    return dJdy


def compute_layer_output_gradient_batch(y: np.ndarray, dJdz: np.ndarray, w: np.ndarray,
                                  activation_function_dif: Callable[[float], float]) -> np.ndarray:
    """ Compute the gradient of the loss function with respect to the output of the layer.
    This function compute the gradient relative to the output of the layer k-1

    Args:
        y: The output of the k-1 layer (ndarray of shape (batch_size, n))
        dJdz: The gradient of the loss function with respect to the output of the k layer (ndarray of shape
            (batch_size, m))
        w: The weights of the k layer (ndarray of shape (n, m))
        activation_function_dif: The derivative of the activation function of the k layer.
    Returns:
        The gradient of the loss function with respect to the output of the k layer (ndarray of shape (1, n))
    """

    batch_size = y.shape[0]
    n = y.shape[1]
    m = w.shape[1]

    h = np.vectorize(activation_function_dif)

    dzdy = w * h(y @ w).reshape((batch_size, 1, m))  # (batch_size, n, m)

    dJdy = dJdz @ dzdy.transpose(0, 2, 1)  # (batch_size, 1, n)
    return dJdy
