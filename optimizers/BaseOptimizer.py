from abc import ABC, abstractmethod
import numpy as np


class BaseOptimizer(ABC):
    @abstractmethod
    def get_weights_delta(self):
        pass

    @abstractmethod
    def update_weights_delta(self, gradient: np.ndarray) -> None:
        pass
