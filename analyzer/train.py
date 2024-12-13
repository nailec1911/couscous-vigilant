"""train a neural network
"""
from sys import stderr
from neural_network.neural_network import NeuralNetwork
from pickle import dump


def train(nn: NeuralNetwork, save: str):

    try:
        with open(save, 'wb') as file:
            dump(nn, file)
    except Exception:
        print(f"Didn't manage to write NN in file {save}", file=stderr)
        return 84
    return 0
