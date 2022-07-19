import numpy as np

from model.model import Model


model = Model.load("trained_models/first.pkl")
X = np.arange(10).reshape((5, 2))
print(f'actual: {(2 * X[:, 0] + 10 * X[:, 1] + 90).reshape((5, 1))}')
print(f'predicted: {model.predict(X)}')
