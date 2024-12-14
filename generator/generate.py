"""generate a neural network
"""
from sys import stderr
from pickle import dump
from generator.config_parsing import Config, Conf_parameters
from neural_network.conv_nn import NeuralNetwork
# from neural_network.neural_network import NeuralNetwork


def generate_nn(filename: str, conf: Conf_parameters) -> None:
    """Generate a neural network and put it in the file

    Args:
        filename (str): name of the resulting file
    """
    print(f"Generating {filename}")

    input_shape = (16, 8, 8)
    conv_layers = [
        {"num_filters": 32, "input_depth": 16, "kernel_size": 3, "eta": 0.01},
        {"num_filters": 64, "input_depth": 32, "kernel_size": 3, "eta": 0.01},
    ]
    fully_connected = [1024, 4]

    nn = NeuralNetwork(input_shape, conv_layers, fully_connected, eta=0.001)

    # nn = NeuralNetwork(conf.layers, conf.epoch, conf.eta)
    try:
        with open(filename, 'wb') as file:
            dump(nn, file)
    except Exception:
        print(f"Didn't manage to write NN in file {filename}", file=stderr)
    return


def generate_files(conf: Config) -> None:
    """generate neurals network from a config

    Args:
        conf (Config): configuration to use
    """
    names = conf.get_names()
    print(f"The files {names} will be generated")
    for file in names:
        generate_nn(file, conf.conf)
    return
