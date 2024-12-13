"""predict a move for chess
"""
from typing import List
from analyzer.board_parsing import Board
from neural_network.neural_network import NeuralNetwork


def predict(nn: NeuralNetwork, inputs: List[Board]) -> int:
    """Get the results of a neural network for a list of boards

    Args:
        nn (NeuralNetwork): neural network to use
        inputs (List[Board]): boards to compute

    Returns:
        int: error code
    """
    for board in inputs:
        nn.forward(board.boards)
        # TODO compute properly
    return 0
