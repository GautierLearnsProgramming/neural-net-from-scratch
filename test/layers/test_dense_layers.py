from layers.DenseLayer import DenseLayer
import numpy as np


def test_dense_layer_forward():
    dense_layer = DenseLayer(input_size=2, output_size=4)
    weights = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [100, 200, 300, 400]])

    dense_layer.load_weights(weights)
    input_data = np.array([[1, 2], [3, 4], [5, 6]])
    assert np.array_equal(dense_layer.forward(input_data), np.array([[111, 214, 317, 420], [123, 230, 337, 444],
                                                                     [135, 246, 357, 468]]))
