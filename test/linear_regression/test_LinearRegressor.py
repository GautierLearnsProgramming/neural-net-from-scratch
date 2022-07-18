import numpy as np

from linear_regression.LinearRegressor import LinearRegressor, Batching
from error import mse


def test_BatchLinearRegressor():
    amount = 500
    dimension = 2
    X = np.random.rand(amount * dimension).reshape((amount, dimension))
    y = (2 * X[:, 0] + 10 * X[:, 1] + 90).reshape((amount, 1))

    lin_reg = LinearRegressor()
    lin_reg.fit(X, y)
    pred = lin_reg.predict(X)

    assert mse.compute_mse(pred, y) < 1e-5

    mini_batch_lin_reg = LinearRegressor(batching=Batching.BATCH, batch_size=50)
    mini_batch_lin_reg.fit(X, y)
    pred = mini_batch_lin_reg.predict(X)

    assert mse.compute_mse(pred, y) < 1e-3
