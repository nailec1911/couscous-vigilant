import numpy as np
from numba import jit, njit
from numba import cuda
from neural_network.convolve2d import convolve2d

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def leaky_relu(x, alpha=0.01):
    return np.maximum(x, alpha * x)

def leaky_relu_derivative(x, alpha=0.01):
    x = np.asarray(x)  # Ensure it's a NumPy array
    derivative = np.ones_like(x)
    derivative[x <= 0] = alpha
    return derivative

def stable_softmax(outputs):
    outputs -= np.max(outputs)
    exp_outputs = np.exp(outputs)
    return exp_outputs / np.sum(exp_outputs)

def categorical_crossentropy(target, outputs, epsilon=1e-12):
    outputs = np.clip(outputs, epsilon, 1 - epsilon)
    return -np.sum(target * np.log(outputs))

def im2col(inputs, kernel_size):
    """
    Rearranges the input into a 2D matrix where each column is a flattened sliding window.
    """
    input_depth, input_height, input_width = inputs.shape
    output_height = input_height - kernel_size + 1
    output_width = input_width - kernel_size + 1

    col = np.zeros((input_depth * kernel_size * kernel_size, output_height * output_width))
    col_idx = 0
    for i in range(output_height):
        for j in range(output_width):
            patch = inputs[:, i:i + kernel_size, j:j + kernel_size].reshape(-1)
            col[:, col_idx] = patch
            col_idx += 1
    return col

def pretty_print_prediction(outputs, target):
    print("[", end="")
    for output in outputs:
        print(f" {output:.5f}", end="")
    print(" ]", end="")
    print(" |", target)


class ConvLayer:
    def __init__(self, num_filters, input_depth, kernel_size, eta=0.01):
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.eta = eta
        self.inputs = None
        self.outputs = None
        # He Initialization for kernels
        scale = np.sqrt(2 / (input_depth * kernel_size * kernel_size))
        self.kernels = np.random.randn(num_filters, input_depth, kernel_size, kernel_size) * scale

        # Biases initialized to small random values
        self.biases = np.random.rand(num_filters) * 0.01

    def forward(self, inputs):
        self.inputs = inputs  # Cache inputs for backward pass

        # Calculate output dimensions
        input_depth, input_height, input_width = inputs.shape
        output_height = input_height - self.kernel_size + 1
        output_width = input_width - self.kernel_size + 1

        # Perform im2col to flatten the sliding windows
        col = im2col(inputs, self.kernel_size)  # Shape: (input_depth * kernel_size^2, output_height * output_width)

        # Flatten kernels for matrix multiplication
        filters = self.kernels.reshape(self.num_filters, -1)  # Shape: (num_filters, input_depth * kernel_size^2)

        # Perform the convolution using matrix multiplication
        outputs = np.dot(filters, col) + self.biases[:, None]  # Add biases (broadcasted)

        # Reshape the outputs to (num_filters, output_height, output_width)
        self.outputs = outputs.reshape(self.num_filters, output_height, output_width)

        # Apply activation function
        self.outputs = leaky_relu(self.outputs)
        return self.outputs

    def backward(self, grad_outputs):
        """
        Backpropagation using matrix multiplication.
        """
        grad_outputs = grad_outputs * leaky_relu_derivative(self.outputs)  # Apply derivative of activation function

        # Compute gradients w.r.t. biases
        grad_biases = np.sum(grad_outputs, axis=(1, 2))

        # Compute gradients w.r.t. kernels
        input_depth, input_height, input_width = self.inputs.shape
        grad_kernels = np.zeros_like(self.kernels)
        grad_inputs = np.zeros_like(self.inputs)

        # Flatten inputs and gradients for matrix operations
        col = im2col(self.inputs, self.kernel_size)  # Shape: (input_depth * kernel_size^2, output_height * output_width)
        grad_outputs_reshaped = grad_outputs.reshape(self.num_filters, -1)  # Shape: (num_filters, output_height * output_width)

        # Gradient w.r.t. kernels (matrix multiplication)
        grad_kernels_flat = np.dot(grad_outputs_reshaped, col.T)  # Shape: (num_filters, input_depth * kernel_size^2)
        grad_kernels = grad_kernels_flat.reshape(self.kernels.shape)  # Reshape to (num_filters, input_depth, kernel_size, kernel_size)

        # Gradient w.r.t. inputs (transposed matrix multiplication)
        filters_flat = self.kernels.reshape(self.num_filters, -1)  # Shape: (num_filters, input_depth * kernel_size^2)
        grad_inputs_col = np.dot(filters_flat.T, grad_outputs_reshaped)  # Shape: (input_depth * kernel_size^2, output_height * output_width)

        # Reshape col back to the input gradient shape
        output_height = grad_outputs.shape[1]
        output_width = grad_outputs.shape[2]
        for i in range(output_height):
            for j in range(output_width):
                patch = grad_inputs_col[:, i * output_width + j].reshape(self.inputs.shape[0], self.kernel_size, self.kernel_size)
                grad_inputs[:, i:i + self.kernel_size, j:j + self.kernel_size] += patch

        # Update weights and biases
        self.kernels -= self.eta * grad_kernels
        self.biases -= self.eta * grad_biases

        return grad_inputs


class Perceptron:
    def __init__(self, input_nbr, eta=0.01):
        self.eta = eta

        # He Initialization for weights
        scale = np.sqrt(2 / input_nbr)
        self.weights = np.random.randn(input_nbr) * scale

        # Bias initialized to a small random value
        self.bias = np.random.rand() * 0.01

    def predict(self, inputs):
        weighted_sum = np.dot(inputs, self.weights) + self.bias
        return leaky_relu(weighted_sum)

    def update_weights(self, inputs, delta):
        self.weights += self.eta * delta * inputs
        self.bias += self.eta * delta

class Layer:
    def __init__(self, nbr_neurons, input_size, eta=0.01):
        self.neurons = [Perceptron(input_size, eta) for _ in range(nbr_neurons)]
        self.outputs = np.zeros(nbr_neurons)

    def forward(self, inputs):
        self.inputs = inputs
        self.outputs = np.array([neuron.predict(inputs) for neuron in self.neurons])
        return self.outputs

    def backward(self, inputs, deltas):
        new_deltas = np.zeros(len(inputs))
        for i, neuron in enumerate(self.neurons):
            delta = deltas[i] * leaky_relu_derivative(self.outputs[i])
            neuron.update_weights(inputs, delta)
            new_deltas += delta * neuron.weights
        return new_deltas

class NeuralNetwork:
    def __init__(self, input_shape, conv_layers, fully_connected, eta=0.01, epoch=1):
        # for now the four outputs are hard coded.
        # it is possible to let the user choose but idk if it is logic
        self.conv_layers = [ConvLayer(**params) for params in conv_layers]
        self.fc_layers = [
            Layer(size, prev_size, eta)
            for size, prev_size in zip(fully_connected[1:], fully_connected[:-1])
        ]
        self.output_layer = Layer(4, fully_connected[-1], eta)
        self.epoch = epoch

    def forward(self, inputs):
        for layer in self.conv_layers:
            inputs = layer.forward(inputs)
        inputs = inputs.flatten()
        for layer in self.fc_layers:
            inputs = layer.forward(inputs)
        outputs = self.output_layer.forward(inputs)
        outputs = stable_softmax(outputs)
        return outputs

    def train(self, inputs, target):
        outputs = self.forward(inputs)
        pretty_print_prediction(outputs, target)
        loss = categorical_crossentropy(target, outputs)
        grad_outputs = outputs - target
        grad_inputs = self.output_layer.backward(self.fc_layers[-1].outputs, grad_outputs)
        for i in range(len(self.fc_layers) - 1, 0, -1):
            grad_inputs = self.fc_layers[i].backward(self.fc_layers[i - 1].outputs, grad_inputs)
        grad_inputs = self.fc_layers[0].backward(inputs.flatten(), grad_inputs)
        for layer in reversed(self.conv_layers):
            grad_inputs = layer.backward(grad_inputs.reshape(layer.outputs.shape))
        return loss


# Example of usage
if __name__ == '__main__':
    input_shape = (13, 8, 8)
    conv_layers = [
        {"num_filters": 26, "input_depth": 13, "kernel_size": 3, "eta": 0.1},
        {"num_filters": 52, "input_depth": 26, "kernel_size": 3, "eta": 0.1},
    ]
    fully_connected = [832, 512]

    nn = NeuralNetwork(input_shape, conv_layers, fully_connected, eta=0.01)

    dataset = [
        (np.random.rand(13, 8, 8), [1, 0, 0, 0]),  # Checkmate
        (np.random.rand(13, 8, 8), [0, 1, 0, 0]),  # Check
        (np.random.rand(13, 8, 8), [0, 0, 1, 0]),  # Pat
        (np.random.rand(13, 8, 8), [0, 0, 0, 1]),  # Nothing
    ]

    epochs = 300
    for epoch in range(epochs):
        total_loss = 0
        for inputs, target in dataset:
            inputs = np.array(inputs)
            target = np.array(target)
            loss = nn.train(inputs, target)
            total_loss += loss
        print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}")
