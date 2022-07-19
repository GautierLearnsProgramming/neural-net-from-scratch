from typing import List
import numpy as np
import cost.mse as mse
import pickle

from layers.BaseLayer import BaseLayer
from batch.BaseBatcher import BaseBatcher
from batch.MiniBatcher import MiniBatcher


class Model:
    def __init__(self, cost: str = "mse", batcher: BaseBatcher = None):
        self.output = None
        self.layers: List[BaseLayer] = []
        self.eval = False
        if cost == "mse":
            self.cost_function = mse.mse
            self.cost_gradient = mse.mse_gradient
        if batcher is None:
            self.batcher = MiniBatcher()

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._forward(X)

    def add_layer(self, layer):
        self.layers.append(layer)

    def _forward(self, input_data: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            input_data = layer.forward(input_data)
        self.output = input_data
        return input_data

    def _backward(self, y_pred: np.ndarray, y: np.ndarray) -> None:
        output_gradient = self.cost_gradient(y_pred, y)
        for layer in reversed(self.layers):
            output_gradient = layer.backward(output_gradient)

    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 100, verbose: int = 0):
        for epoch in range(epochs):
            total_error = 0
            self.batcher.setData(X)
            for batch in self.batcher:
                y_pred = self._forward(X[batch])
                total_error += self.cost_function(y_pred, y[batch])
                self._backward(y_pred, y[batch])
            if verbose > 0:
                print("Epoch: {}".format(epoch))
                print("Total error: {}".format(total_error))

    def eval(self):
        self.eval = True
        for layer in self.layers:
            layer.eval()

    def save(self, filename: str):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename: str):
        with open(filename, 'rb') as f:
            return pickle.load(f)

