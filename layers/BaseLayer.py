from abc import ABC, abstractmethod
import numpy as np


class BaseLayer(ABC):
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape

    @abstractmethod
    def forward(self, input_data: np.ndarray) -> np.ndarray:
        pass
