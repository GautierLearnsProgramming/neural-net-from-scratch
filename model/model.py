from typing import List
import numpy as np

from layers.BaseLayer import BaseLayer


class Model:
    def __init__(self):
        self.layers: List[BaseLayer] = []
        self.eval = False

    def add_layer(self, layer):
        self.layers.append(layer)

    def forward(self, input_data: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            input_data = layer.forward(input_data)
        return input_data

    def eval(self):
        self.eval = True
