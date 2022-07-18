from abc import ABC, abstractmethod
import numpy as np


class BaseBatcher(ABC):
    """
    Base class for batchers
    """
    @abstractmethod
    def __bool__(self):
        pass

    @abstractmethod
    def __iter__(self):
        pass

    @abstractmethod
    def __next__(self):
        pass

    @abstractmethod
    def setData(self, data: np.ndarray):
        pass
