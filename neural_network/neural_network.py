"""class for neural network
"""
from typing import List


class NeuralNetwork:
    """neural network class
    """

    def __init__(self, layers: List, epoch: int, eta: float):
        self.layers = layers
        self.epoch = epoch

    def forward(self, inputs):
        print("forward")

    def train(self, inputs, expected):
        print(self.layers, expected)
