from typing import List
import numpy as np
import cost.mse as mse
import cost.log_loss as log_loss
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
        elif cost == "log_loss":
            self.cost_function = log_loss.log_loss
            self.cost_gradient = log_loss.log_loss_gradient
        else:
            raise ValueError("Invalid cost function")

        if batcher is None:
            self.batcher = MiniBatcher()

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._forward(X)

    def add_layer(self, layer):
        self.layers.append(layer)

    def _forward(self, input_data: np.ndarray, verbose=0) -> np.ndarray:
        for layer in self.layers:
            input_data = layer.forward(input_data)
            if verbose >= 5:
                print("Layer Output: {}".format(input_data))
        self.output = input_data
        return input_data

    def _backward(self, y_pred: np.ndarray, y: np.ndarray, verbose=0) -> None:
        output_gradient = self.cost_gradient(y_pred, y)
        if verbose >= 10:
            print("Output gradient: {}".format(output_gradient))
        for layer in reversed(self.layers):
            output_gradient = layer.backward(output_gradient)
            if verbose >= 10:
              print("Layer gradient: {}".format(output_gradient))

    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 100, verbose: int = 0):
        """ Train the model.

        Args:
            X: the input data (ndarray of shape (sample_size, n))
            y: the target data (ndarray of shape (sample_size, n))
        """
        for epoch in range(epochs):
            total_error = 0
            self.batcher.setData(X)
            for batch in self.batcher:
                y_pred = self._forward(X[batch], verbose)
                total_error += self.cost_function(y_pred, y[batch])
                self._backward(y_pred, y[batch], verbose)
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

