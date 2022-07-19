import numpy as np


def logistic(x):
    return 1 / (1 + np.exp(-x))


def diff_logistic(x):
    return np.exp(np.minimum(x, 30)) / (np.exp(np.minimum(x, 30)) + 1) ** 2
