"""train a neural network
"""
from sys import stderr
from typing import List
from pickle import dump

import numpy as np
from analyzer.board_parsing import Board
from neural_network.conv_nn import NeuralNetwork


def train(nn: NeuralNetwork, save: str, inputs: List[Board]) -> int:
    """train a neural network from a dataset

    Args:
        nn (NeuralNetwork): neural network to use
        save (str): filepath to load the result
        inputs (List[Board]): list of board to train on

    Returns:
        int: error code
    """

    epoch = 0
    for epoch in range(nn.epoch):
        total_loss = 0
        for board in inputs:
            target = board.expected
            board = np.array(board.boards)
            total_loss += nn.train(board, target)
        print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}")

    try:
        with open(save, 'wb') as file:
            dump(nn, file)
    except Exception:
        print(f"Didn't manage to write NN in file {save}", file=stderr)
        return 84
    print(f"New neural network saved in {save}")
    return 0
