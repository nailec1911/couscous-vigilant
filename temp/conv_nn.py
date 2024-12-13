import numpy as np

PLANE_NBR = 16
BOARD_SIZE = 8

class ConvLayer:
    def __init__(self, num_filters, input_depth, kernel_size, eta=0.01):
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.eta = eta
        # Initialize kernels (filters) randomly
        self.kernels = np.random.rand(num_filters, input_depth, kernel_size, kernel_size)
        self.biases = np.random.rand(num_filters)

    def forward(self, inputs):
        self.inputs = inputs
        # Perform convolution
        self.outputs = []
        for k in range(self.num_filters):
            output = np.zeros((inputs.shape[1] - self.kernel_size + 1, inputs.shape[2] - self.kernel_size + 1))
            for d in range(inputs.shape[0]):
                output += self.convolve2d(inputs[d], self.kernels[k, d])
            output += self.biases[k]
            self.outputs.append(sigmoid(output))  # Apply activation
        return np.array(self.outputs)

    def convolve2d(self, input_plane, kernel):
        # Simple 2D convolution (no padding)
        kernel_height, kernel_width = kernel.shape
        output_height = input_plane.shape[0] - kernel_height + 1
        output_width = input_plane.shape[1] - kernel_width + 1
        output = np.zeros((output_height, output_width))
        for i in range(output_height):
            for j in range(output_width):
                region = input_plane[i:i+kernel_height, j:j+kernel_width]
                output[i, j] = np.sum(region * kernel)
        return output

    def backward(self, grad_outputs):
        # Implement backpropagation (left as an exercise, as it's lengthy to detail here)
        pass

class NeuralNetwork:
    def __init__(self, input_shape, conv_layers, fully_connected, eta=0.01):
        self.conv_layers = [ConvLayer(**params) for params in conv_layers]
        self.fc_layers = [Layer(size, prev_size, eta) for size, prev_size in zip(fully_connected[1:], fully_connected[:-1])]
        self.policy_head = Layer(4672, fully_connected[-1], eta)  # For move probabilities
        self.value_head = Layer(1, fully_connected[-1], eta)     # For scalar evaluation

    def forward(self, inputs):
        for layer in self.conv_layers:
            inputs = layer.forward(inputs)
        inputs = inputs.flatten()  # Flatten for fully connected layers
        for layer in self.fc_layers:
            inputs = layer.forward(inputs)
        policy = self.policy_head.forward(inputs)
        value = self.value_head.forward(inputs)
        return policy, value

    def train(self, inputs, policy_target, value_target):
        policy, value = self.forward(inputs)
        # Compute loss for both heads
        policy_loss = policy_target - policy
        value_loss = value_target - value
        # Backpropagate losses through the network
        # (implement backward for conv_layers and fully connected layers)


###########################################################

def encode_chessboard(board_state):
    encoded = np.zeros((PLANE_NBR, BOARD_SIZE, BOARD_SIZE))  # Assuming 16 feature planes
    print(encoded)
    # Example: Fill planes based on board_state (this is a placeholder)
    # Plane 0: White pawns
    # Plane 1: Black pawns
    # Plane 14: Castling rights
    # Plane 15: Side to move
    # ... Fill the planes as needed
    return encoded

if __name__ == '__main__':
    input_shape = (PLANE_NBR, BOARD_SIZE, BOARD_SIZE)
    
    conv_layers = [
        {"num_filters": 32, "input_depth": 16, "kernel_size": 3},  # First convolutional layer
        {"num_filters": 64, "input_depth": 32, "kernel_size": 3},  # Second convolutional layer
    ]
    
    fully_connected = [1024, 512]  # Example: Fully connected layers with 1024 and 512 neurons

    # Instantiate the neural network
    nn = NeuralNetwork(input_shape, conv_layers, fully_connected, eta=0.1)
    
    encode_chessboard(None)