import numpy as np

from model.model import Model
from layers.DenseLayer import DenseLayer


def test_model():
    model = Model()
    layer1 = DenseLayer(input_shape=2, output_shape=4)
    layer1.load_weights(np.array([[1, 2, 3, 4], [5, 6, 7, 8]]), np.array([100, 200, 300, 400]))

    layer2 = DenseLayer(input_shape=4, output_shape=2)
    layer2.load_weights(np.array([[1, 2], [3, 4], [5, 6], [7, 8]]), np.array([1000, 2000]))
    model.add_layer(layer1)
    model.add_layer(layer2)

    assert np.array_equal(model.forward(np.array([1, 2])), np.array([6278, 8340]))
    assert np.array_equal(model.forward(np.array([[1, 2], [3, 4]])), np.array([[6278, 8340], [6606, 8740]]))
