import numpy as np

class Sequential:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.loss_prime = None

    def add(self, layer): self.layers.append(layer)
    def use(self, loss, loss_prime):
        self.loss = loss
        self.loss_prime = loss_prime

    def predict(self, input_data):
        result = []
        for x in input_data:
            output = x.reshape(-1, 1)
            for layer in self.layers:
                output = layer.forward(output)
            result.append(output.flatten())
        return np.array(result)

    def fit(self, x_train, y_train, epochs, learning_rate):
        errors = []
        for epoch in range(epochs):
            error = 0
            for i in range(len(x_train)):
                output = x_train[i].reshape(-1, 1)
                for layer in self.layers:
                    output = layer.forward(output)

                error += self.loss(y_train[i], output)
                grad = self.loss_prime(y_train[i], output)

                for layer in reversed(self.layers):
                    grad = layer.backward(grad, learning_rate)

            error /= len(x_train)
            errors.append(error)
            if (epoch + 1) % 100 == 0 or epoch == 1:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {error:.4f}")
        return errors
