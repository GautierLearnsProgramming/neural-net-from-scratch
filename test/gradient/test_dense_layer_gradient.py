import numpy as np

from gradient.dense_layer_output_gradient import compute_layer_output_gradient_batch
from gradient.dense_layer_weight_gradient import compute_dense_layer_weight_gradient_batch
from activation.relu import diff_relu


def test_compute_layer_output_gradient_batch():
    y = np.array([[2, 1]])
    y_2 = np.array([[2, 1], [2, 1]])
    w = np.array([[2, 3], [-4, 4]])
    next_layer_gradient = np.array([[-4, 8]])
    assert np.array_equal(compute_layer_output_gradient_batch(y, next_layer_gradient, w, diff_relu),
                          np.array([[24, 32]]))
    next_layer_gradient = np.array([[-4, 8], [-4, 8]])
    assert np.array_equal(compute_layer_output_gradient_batch(y_2, next_layer_gradient, w, diff_relu),
                          np.array([[24, 32], [24, 32]]))


def test_computer_layer_weight_gradient_batch():
    y = np.array([[2, 1]])
    y_2 = np.array([[2, 1], [2, 1]])
    w = np.array([[2, 3], [-4, 4]])
    next_layer_gradient = np.array([[-4, 8]])
    assert np.array_equal(compute_dense_layer_weight_gradient_batch(y, next_layer_gradient, w, diff_relu),
                          np.array([[0, 16], [0, 8]]))

    next_layer_gradient = np.array([[-4, 8], [-4, 8]])
    assert np.array_equal(compute_dense_layer_weight_gradient_batch(y_2, next_layer_gradient, w, diff_relu),
                          np.array([[0, 32], [0, 16]]))
