from layers.DenseLayer import DenseLayer
import numpy as np


def test_dense_layer_forward():
    dense_layer = DenseLayer(input_shape=2, output_shape=4)
    weights = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    bias = np.array([100, 200, 300, 400])

    dense_layer.load_weights(weights, bias)
    input_data = np.array([[1, 2], [3, 4], [5, 6]])
    assert np.array_equal(dense_layer.forward(input_data), np.array([[111, 214, 317, 420], [123, 230, 337, 444],
                                                                     [135, 246, 357, 468]]))
