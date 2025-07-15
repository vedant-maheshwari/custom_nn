import numpy as np

class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, input): pass
    def backward(self, output_gradient, learning_rate): pass

class Dense(Layer):
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(output_size, input_size) * np.sqrt(2. / input_size)
        self.bias = np.zeros((output_size, 1))

    def forward(self, input):
        self.input = input
        self.output = np.dot(self.weights, self.input) + self.bias
        return self.output

    def backward(self, output_gradient, learning_rate):
        weight_gradient = np.dot(output_gradient, self.input.T)
        bias_gradient = np.sum(output_gradient, axis=1, keepdims=True)

        # np.clip(weight_gradient, -1, 1, out=weight_gradient)
        # np.clip(bias_gradient, -1, 1, out=bias_gradient)

        self.weights -= learning_rate * weight_gradient
        self.bias -= learning_rate * bias_gradient

        return np.dot(self.weights.T, output_gradient)
