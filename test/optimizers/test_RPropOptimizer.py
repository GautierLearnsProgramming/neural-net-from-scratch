import numpy as np

from optimizers.RPropOptimizer import RPropOptimizer


def test_RPropOptimizer():
    optim = RPropOptimizer(step_size_shrink=0.5, step_size_grow=2, step_size_init=1)
    optim.update_learning_rates(np.array([1, 2, 3]))
    assert np.array_equal(optim.get_learning_rates(), np.array([-1, -1, -1]))
    optim.update_learning_rates(np.array([1, 2, -3]))
    assert np.array_equal(optim.get_learning_rates(), np.array([-2, -2, 0.5]))
