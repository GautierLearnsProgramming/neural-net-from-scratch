import numpy as np
from layers.BaseLayer import BaseLayer
from typing import Callable
from activation.relu import relu


class DenseLayer(BaseLayer):
    def __init__(self, input_shape, output_shape, activation_function: Callable[[float], float] = None):
        super().__init__(input_shape, output_shape)
        self.output = None
        self.weights = np.random.randn(input_shape, output_shape)
        self.bias = np.random.randn(output_shape)
        if activation_function is None:
            self.activation_function = np.vectorize(relu)
        else:
            self.activation_function = np.vectorize(activation_function)

    def forward(self, input_data: np.ndarray) -> np.ndarray:
        output = self.activation_function(np.dot(input_data, self.weights) + self.bias)
        if not self.eval_mode:
            self.output = output
        return output

    def backward(self, next_layer_gradient: np.ndarray) -> np.ndarray:
        pass

    def load_weights(self, weights: np.ndarray, bias: np.ndarray):
        self.weights = weights
        self.bias = bias
