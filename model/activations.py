import numpy as np
from scipy.special import expit

class Activation:
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime

    def forward(self, input):
        self.input = input
        return self.activation(input)

    def backward(self, output_gradient, learning_rate):
        return output_gradient * self.activation_prime(self.input)

class Relu(Activation):
    def __init__(self):
        super().__init__(
            lambda x: np.maximum(0, x),
            lambda x: np.where(x > 0, 1, 0)
        )

# class Sigmoid(Activation):
#     def __init__(self):
#         super().__init__(
#             lambda x: expit(x),
#             lambda x: expit(x) * (1 - expit(x))
#         )

class Sigmoid(Activation):
    def __init__(self):
        def sigmoid(x):
            # Prevent overflow with a stable sigmoid
            return np.where(
                x >= 0,
                1 / (1 + np.exp(-x)),
                np.exp(x) / (1 + np.exp(x))
            )

        def sigmoid_prime(x):
            s = sigmoid(x)
            return s * (1 - s)

        super().__init__(sigmoid, sigmoid_prime)