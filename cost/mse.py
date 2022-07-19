import numpy as np


def mse(x, y):
    return np.mean((x - y) ** 2)


def mse_gradient(x, y):
    return 2 * (x - y) / x.shape[0]
