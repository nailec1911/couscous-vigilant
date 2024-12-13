"""train a neural network
"""
from sys import stderr
from typing import List
from pickle import dump
from analyzer.board_parsing import Board
from neural_network.neural_network import NeuralNetwork


def train(nn: NeuralNetwork, save: str, inputs: List[Board]) -> int:
    """train a neural network from a dataset

    Args:
        nn (NeuralNetwork): neural network to use
        save (str): filepath to load the result
        inputs (List[Board]): list of board to train on

    Returns:
        int: error code
    """

    for board in inputs:
        nn.train(board.boards, board.expected)
        # TODO train properly

    try:
        with open(save, 'wb') as file:
            dump(nn, file)
    except Exception:
        print(f"Didn't manage to write NN in file {save}", file=stderr)
        return 84
    return 0
