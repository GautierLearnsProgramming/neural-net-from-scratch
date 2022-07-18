import numpy as np
from enum import Enum
from typing import Union

from batch.MiniBatcher import MiniBatcher
from batch.FullBatcher import FullBatcher
from optimizers.BaseOptimizer import BaseOptimizer
from optimizers.RPropOptimizer import RPropOptimizer
from optimizers.RMSPropOptimizer import RMSPropOptimizer
from linear_regression import _compute


class Batching(Enum):
    FULL = "FULL"
    BATCH = "BATCH"


class LinearRegressor:
    def __init__(self, epochs: int = 1000, optimizer: BaseOptimizer = None, batching: Union[Batching, str] = Batching.FULL,
                 batch_size: int = 50, seed: int = None):
        self.weights = None
        self.epochs = epochs
        self.optimizer = optimizer
        if optimizer is None:
            if batching == Batching.FULL:
                self.optimizer = RPropOptimizer()
            elif batching == Batching.BATCH:
                self.optimizer = RMSPropOptimizer()
        if batching == Batching.FULL:
            self.batcher = FullBatcher()
        elif batching == Batching.BATCH:
            self.batcher = MiniBatcher(batch_size, seed)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.weights = np.ones(X.shape[1] + 1).reshape((X.shape[1] + 1, 1))

        for epoch in range(self.epochs):
            self.batcher.setData(X)
            while self.batcher:
                batch_index = self.batcher.__next__()
                gradient = _compute.compute_mse_gradient(X[batch_index], y[batch_index], self.weights)
                self.optimizer.update_weights_delta(gradient)
                self.weights += self.optimizer.get_weights_delta()

    def get_weights(self):
        return self.weights

    def predict(self, X):
        return np.c_[np.ones(X.shape[0]), X] @ self.weights
