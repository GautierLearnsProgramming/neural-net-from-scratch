import numpy as np

from batch.BaseBatcher import BaseBatcher


class MiniBatcher(BaseBatcher):
    def __init__(self, batch_size: int = 50, seed: int = None):
        self.length = None
        self.empty = True
        self.batch_size = batch_size
        self.index = 0
        self.permutation = None
        if seed:
            np.random.seed(seed)

    def __bool__(self):
        return not self.empty

    def __iter__(self):
        return self

    def __next__(self):
        if not self.empty:
            mini_batch = self.permutation[self.index:self.index + self.batch_size]
            self.index += self.batch_size
            if self.index >= self.length:
                self.empty = True
            return mini_batch
        else:
            raise StopIteration

    def setData(self, data: np.ndarray):
        self.empty = False
        self.index = 0
        self.length = len(data)
        self.permutation = np.random.permutation(len(data))
