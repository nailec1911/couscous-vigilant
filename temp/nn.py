import numpy as np

DATASET = [
    [[0, 0], [0]],
    [[0, 1], [1]],
    [[1, 0], [1]],
    [[1, 1], [0]],
]
# Activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)
# -----------------------------------------------

class Perceptron:
    def __init__(self, input_nbr, eta=0.01):
        print(f"Created perceptron with {input_nbr} inputs.")
        self.eta = eta
        self.weights = np.random.rand(input_nbr)
        self.bias = np.random.rand()

    def predict(self, inputs):
        weighted_sum = np.dot(inputs, self.weights) + self.bias
        return sigmoid(weighted_sum)

    def update_weights(self, inputs, delta):
        # Update weights and bias based on the delta error
        for i, _ in  enumerate(self.weights):
            self.weights[i] += self.eta * delta * inputs[i]

        self.bias += self.eta * delta

class Layer:
    def __init__(self, nbr_neurons, input_size, eta=0.01):
        self.neurons = [Perceptron(input_size, eta) for _ in range(nbr_neurons)]
        self.outputs = np.zeros(nbr_neurons)

    def forward(self, inputs):
        self.outputs = np.array([neuron.predict(inputs) for neuron in self.neurons])
        return self.outputs

    def backward(self, inputs, deltas):
        new_deltas = np.zeros(len(inputs))
        for i, neuron in enumerate(self.neurons):
            delta = deltas[i] * sigmoid_derivative(self.outputs[i])
            neuron.update_weights(inputs, delta)
            new_deltas += delta * neuron.weights
        return new_deltas

class NeuralNetwork:
    def __init__(self, layer_sizes, eta=0.01):
        self.layers = []
        for i in range(1, len(layer_sizes)):
            self.layers.append(Layer(layer_sizes[i], layer_sizes[i - 1], eta))

    def forward(self, inputs):
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs

    def train(self, inputs, target_outputs):
        outputs = self.forward(inputs)

        # error calculation
        deltas = target_outputs - outputs

        for i in reversed(range(len(self.layers))):
            deltas = self.layers[i].backward(inputs if i == 0 else self.layers[i - 1].outputs, deltas)


if __name__ == '__main__':
    nn = NeuralNetwork([2, 4, 1], eta=0.1)

    epochs = 10000
    for _ in range(epochs):
        for cinput, target in DATASET:
            nn.train(np.array(cinput), np.array(target))

    for cinput, target in DATASET:
        prediction = nn.forward(np.array(cinput))
        print(f"Input: {cinput}, Target: {target}, Prediction: {prediction}")
