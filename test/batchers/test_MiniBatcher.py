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

    batcher.setData(data)
    permutation = np.array([0, 1, 8, 5, 3, 4, 7, 9, 6, 2])
    for index, batch in enumerate(batcher):
        if index == 0:
            assert np.array_equal(batch, permutation[:4])
        elif index == 1:
            assert np.array_equal(batch, permutation[4:8])
        elif index == 2:
            assert np.array_equal(batch, permutation[8:])
