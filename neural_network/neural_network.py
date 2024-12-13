"""class for neural network
"""
from typing import List


class NeuralNetwork:
    """neural network class
    """

    def __init__(self, layers: List, epoch: int, eta: float):
        self.layers = layers
        self.epoch = epoch
        pass

    def forward(self, inputs):
        pass

    def train(self, inputs):
        print(self.layers)
        pass
