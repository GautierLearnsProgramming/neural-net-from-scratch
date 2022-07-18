import pytest
import numpy as np

from batch.FullBatcher import FullBatcher


def test_FullBatcher():
    batcher = FullBatcher()
    data = np.arange(20).reshape((10, 2))
    batcher.setData(data)
    assert batcher.__bool__() is True
    assert np.array_equal(batcher.__next__(), data)
    assert batcher.__bool__() is False
    with pytest.raises(StopIteration):
        batcher.__next__()
