import numpy as np

from gradient.dense_layer_output_gradient import compute_layer_output_gradient_batch
from activation.relu import diff_relu


def test_compute_layer_output_gradient_batch():
    y = np.array([[2, 1]])
    y_2 = np.array([[2, 1], [2, 1]])
    w = np.array([[2, 3], [-4, 4]])
    next_layer_gradient = np.array([[-4, 8]])
    assert np.array_equal(compute_layer_output_gradient_batch(y, next_layer_gradient, w, diff_relu),
                          np.array([[[24, 32]]]))
    assert np.array_equal(compute_layer_output_gradient_batch(y_2, next_layer_gradient, w, diff_relu),
                          np.array([[[24, 32]], [[24, 32]]]))
