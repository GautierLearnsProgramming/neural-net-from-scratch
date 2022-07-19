import numpy as np


def squared_error(z: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.sum((z - y) ** 2)


def compute_se_gradient(z: np.ndarray, y: np.ndarray) -> np.ndarray:
    return 2 * (z - y)
