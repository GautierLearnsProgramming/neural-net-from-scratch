import numpy as np

from optimizers.BaseOptimizer import BaseOptimizer


class RPropOptimizer(BaseOptimizer):
    def __init__(self, step_size_shrink: float = 0.5, step_size_grow: float = 1.2, step_size_min: float = 1e-9,
                 step_size_max: float = 50, step_size_init: float = 0.01):
        self.step_size_shrink = step_size_shrink
        self.step_size_grow = step_size_grow
        self.step_size_min = step_size_min
        self.step_size_max = step_size_max
        self.step_size_init = step_size_init
        self.previous_gradient = None
        self.step_size = None

    def get_weights_delta(self) -> np.ndarray:
        return - self.step_size * np.sign(self.previous_gradient)

    def update_weights_delta(self, gradient: np.ndarray) -> None:
        if self.previous_gradient is None:
            # First call of the function
            self.step_size = self.step_size_init * np.ones(gradient.shape)
            self.previous_gradient = gradient
            return

        # Update the step sizes
        for index, grad_product in np.ndenumerate(gradient * self.previous_gradient):
            if grad_product > 0:
                self.step_size[index] = min(self.step_size_grow * self.step_size[index], self.step_size_max)
            elif grad_product < 0:
                self.step_size[index] = max(self.step_size_shrink * self.step_size[index], self.step_size_min)

        self.previous_gradient = gradient
        return

    def get_step_size_shrink(self):
        return self.step_size_shrink

    def get_step_size_grow(self):
        return self.step_size_grow
