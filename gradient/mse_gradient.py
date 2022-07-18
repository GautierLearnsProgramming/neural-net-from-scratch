import numpy as np


def compute_se_gradient(z: np.ndarray, y: np.ndarray) -> np.ndarray:
    return 2 * (z - y)
