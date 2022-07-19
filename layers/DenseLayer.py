import numpy as np
from layers.BaseLayer import BaseLayer
from typing import Callable
from activation.relu import relu, diff_relu
from optimizers.BaseOptimizer import BaseOptimizer
from optimizers.RMSPropOptimizer import RMSPropOptimizer
from gradient.dense_layer_weight_gradient import compute_dense_layer_weight_gradient_batch
from gradient.dense_layer_output_gradient import compute_layer_output_gradient_batch


class DenseLayer(BaseLayer):
    """ DenseLayer class representing a dense layer of a neural network.

    Attributes:
        input_size: The number of input nodes.
        output_size: The number of output nodes.
        weights: The weights of the layer (shape (input_size, output_size)).
        bias: The bias of the layer (shape (output_size)).
        activation_function: The activation function of the layer.
        activation_function_dif: The derivative of the activation function of the layer.
    """
    def __init__(self, input_size, output_size, activation_function: Callable[[float], float] = None,
                 activation_function_dif: Callable[[float], float] = None, optimizer: BaseOptimizer = None):
        super().__init__(input_size, output_size)
        self.input = None
        self.weights = np.random.uniform(0, 1, (input_size + 1, output_size))
        if activation_function is None:
            self.activation_function = np.vectorize(relu)
            self.activation_function_dif = np.vectorize(diff_relu)
        else:
            self.activation_function = np.vectorize(activation_function)
            if activation_function_dif is None:
                raise ValueError("The derivative of the activation function must be provided.")
            self.activation_function_dif = np.vectorize(activation_function_dif)
        if optimizer is None:
            self.optimizer = RMSPropOptimizer()

    def forward(self, input_data: np.ndarray, verbose = 0) -> np.ndarray:
        """ Forward pass of the layer.

        Args:
            input_data: The input data of the layer (ndarray of shape (batch_size, input_size)).
            verbose: The verbosity level of the layer.
        Returns:
            The output of the layer (ndarray of shape (batch_size, output_size)).
        """
        input_data = np.append(input_data, np.ones((input_data.shape[0], 1)), axis=1)

        if verbose > 0:
            print(f'input_data: {input_data}')
            print(f'input_data.shape: {input_data.shape}')
            print(f'self.weights: {self.weights}')

        output = self.activation_function(np.dot(input_data, self.weights))
        if not self.eval_mode:
            self.input = input_data
        return output

    def backward(self, next_layer_gradient: np.ndarray) -> np.ndarray:
        """ Backward pass of the layer.

        Args:
            next_layer_gradient: The gradient of the loss function with respect to the output of the layer (ndarray of
                shape (batch_size, output_size)).
        """
        # print(f'next_layer_gradient_shape: \n{next_layer_gradient.shape}')

        weight_gradient = compute_dense_layer_weight_gradient_batch(self.input, next_layer_gradient, self.weights,
                                                                    self.activation_function_dif)
        output_gradient = compute_layer_output_gradient_batch(self.input[:, :-1], next_layer_gradient, self.weights[:-1, :],
                                                              self.activation_function_dif)
        self.optimizer.update_weights_delta(weight_gradient)
        self.weights += self.optimizer.get_weights_delta()
        return output_gradient

    def load_weights(self, weights: np.ndarray):
        self.weights = weights
