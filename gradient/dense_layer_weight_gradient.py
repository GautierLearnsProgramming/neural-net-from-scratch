import numpy as np
from typing import Callable


def compute_dense_layer_gradient(y: np.ndarray, w: np.ndarray, next_layer_gradient: np.ndarray,
                                 activation_function_dif: Callable[[float], float]) -> np.ndarray:
    """ Compute the gradient of the loss function with respect to the weights of the dense layer.

    This function compute the gradient relative to the weights of the dense layer k, connecting the k-1 layer to the k
    layer.

    Args:
        y: The output of the k-1 layer (ndarray of shape (1, n))
        w: The weights of the k layer (ndarray of shape (n, m))
        next_layer_gradient: The gradient of the loss function with respect to the output of the k layer (ndarray of
            shape (1, m))
        activation_function_dif: The derivative of the activation function of the k layer.
    Returns:
        The gradient of the loss function with respect to the weights of the k layer (ndarray of shape (n, m))
    """
    h = np.vectorize(activation_function_dif)
    dzdw = (h(y @ w).T @ y)  # (m, n)
    dJdw = dzdw.T * next_layer_gradient  # (n, m)
    return dJdw


def compute_dense_layer_weight_gradient_batch(y: np.ndarray, next_layer_gradient: np.ndarray, w: np.ndarray,
                                              activation_function_dif: Callable[[float], float],
                                              verbose: int = 0) -> np.ndarray:
    """ Compute the gradient of the loss function with respect to the weights of the dense layer.

    This function compute the gradient relative to the weights of the dense layer k, connecting the k-1 layer to the k
    layer.

    Args:
        y: The output of the k-1 layer (ndarray of shape (batch_size, n))
        w: The weights of the k layer (ndarray of shape (n, m))
        next_layer_gradient: The gradient of the loss function with respect to the output of the k layer (ndarray of
            shape (batch_size, m))
        activation_function_dif: The derivative of the activation function of the k layer.
        verbose: Verbosity level
    Returns:
        The gradient of the loss function with respect to the weights of the k-1 layer (ndarray of shape (n, m))
    """
    h = np.vectorize(activation_function_dif)

    batch_size = y.shape[0]
    m = w.shape[1]
    n = y.shape[1]

    y = y.reshape((batch_size, 1, n))

    dzdw = (h(y @ w).transpose(0, 2, 1) @ y)  # (m, n)

    dJdw = dzdw.transpose(0, 2, 1) * next_layer_gradient.reshape((batch_size, 1, m))  # (batch_size, n, m)

    if verbose > 0:
        print()
        print(f'y: \n{y}')
        print(f'next_layer_gradient: \n{next_layer_gradient}')
        print(f'w: \n{w}')
        print(f'h(y @ w).T: \n{h(y @ w).transpose(0, 2, 1)}')
        print(f'h(y @ w).T @ y: \n{h(y @ w).transpose(0, 2, 1) @ y}')
        print(f'dJdw: \n{dJdw}')

    return dJdw.sum(axis=0)  # (n, m)
