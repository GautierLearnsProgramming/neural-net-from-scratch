import pytest
import numpy as np

from batch.MiniBatcher import MiniBatcher


def test_MiniBatcher():
    batcher = MiniBatcher(batch_size=4, seed=42)
    data = np.arange(20).reshape((10, 2))
    permutation = np.array([8, 1, 5, 0, 7, 2, 9, 4, 3, 6])
    batcher.setData(data)
    assert batcher.__bool__() is True
    assert np.array_equal(batcher.__next__(), permutation[:4])
    assert batcher.__bool__() is True
    assert np.array_equal(batcher.__next__(), permutation[4:8])
    assert batcher.__bool__() is True
    assert np.array_equal(batcher.__next__(), permutation[8:])
    assert batcher.__bool__() is False
    with pytest.raises(StopIteration):
        batcher.__next__()
