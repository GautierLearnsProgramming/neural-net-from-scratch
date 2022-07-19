import numpy as np


from layers.DenseLayer import DenseLayer
from model.model import Model
import activation.logistic as logistic


model = Model(cost="log_loss")
model.add_layer(DenseLayer(1, 3))
model.add_layer(DenseLayer(3, 3))
model.add_layer(DenseLayer(3, 1, activation_function=logistic.logistic,
                           activation_function_diff=logistic.diff_logistic))


X = np.arange(1000).reshape((1000, 1))
y = np.array([1 * (x > 300) for x in X]).reshape((1000, 1))
X = X / 1000

model.train(X, y, epochs=150, verbose=1)
model.save("../trained_models/log_loss.pkl")
print(model.predict(X))
