from abc import ABC, abstractmethod
import numpy as np


class BaseOptimizer(ABC):
    @abstractmethod
    def get_learning_rates(self):
        pass

    @abstractmethod
    def update_learning_rates(self, gradient: np.ndarray) -> None:
        pass
