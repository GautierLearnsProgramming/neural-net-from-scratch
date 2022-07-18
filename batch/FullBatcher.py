import numpy as np

from batch.BaseBatcher import BaseBatcher


class FullBatcher(BaseBatcher):
    def __init__(self):
        self.length = None
        self.empty = True

    def __iter__(self):
        return self

    def __bool__(self):
        return not self.empty

    def __next__(self):
        if not self.empty:
            self.empty = True
            return np.arange(self.length)
        else:
            raise StopIteration

    def setData(self, data: np.ndarray):
        self.length = len(data)
        self.empty = False
