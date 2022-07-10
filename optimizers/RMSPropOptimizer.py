import numpy as np

from optimizers.BaseOptimizer import BaseOptimizer


class RMSPropOptimizer(BaseOptimizer):
    def __init__(self, decay: float = 0.9, incoming: float = 0.1, learning_rate: float = 0.01):
        self.decay = decay
        self.incoming = incoming
        self.learning_rate = learning_rate
        self.previous_gradient = None
        self.moving_mean_square = None

    def get_weights_delta(self):
        return - self.learning_rate * self.previous_gradient / np.sqrt(self.moving_mean_square)

    def update_weights_delta(self, gradient: np.ndarray) -> None:
        if self.previous_gradient is None:
            # First call of the function
            self.moving_mean_square = self.incoming * np.square(gradient)
            self.previous_gradient = gradient
            return

        self.moving_mean_square = self.decay * self.moving_mean_square + self.incoming * np.square(gradient)
        self.previous_gradient = gradient
