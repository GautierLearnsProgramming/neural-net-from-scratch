import numpy as np

from optimizers.BaseOptimizer import BaseOptimizer
from optimizers.RPropOptimizer import RPropOptimizer


class BatchLinearRegressor:
    def __init__(self, epochs: int = 1000, optimizer: BaseOptimizer = RPropOptimizer()):
        self.weights = None
        self.epochs = epochs
        self.optimizer = optimizer

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.weights = np.ones(X.shape[1] + 1).reshape((X.shape[1] + 1, 1))

        for epoch in range(self.epochs):
            aug_X = np.c_[np.ones(X.shape[0]), X]
            gradient = np.transpose(aug_X) @ ((aug_X @ self.weights).reshape((y.shape[0], y.shape[1])) - y)
            self.optimizer.update_weights_delta(gradient)
            self.weights += self.optimizer.get_weights_delta()

    def get_weights(self):
        return self.weights

    def predict(self, X):
        return np.c_[np.ones(X.shape[0]), X] @ self.weights
