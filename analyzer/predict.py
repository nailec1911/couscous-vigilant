"""predict a move for chess
"""
from typing import List

import numpy as np
from analyzer.board_parsing import Board
from neural_network.neural_network import NeuralNetwork


def interpret_result(result: np.ndarray, turn: str):
    """interpret the neural network result to mak it human readable

    Args:
        result (List[int]): output from the nural network
        turn (str): last player turn
    """
    assert len(result) == 4
    assert turn == 'w' or turn == 'b'
    possibilities = ["Checkmate", "Check", "Stalemate", "Nothing"]
    turns = {'w': 'White', 'b': 'Black'}
    res = np.argmax(result)
    if res < 2:
        print(possibilities[res] + ' ' + turns[turn])
    else:
        print(possibilities[res])


def predict(nn: NeuralNetwork, inputs: List[Board]) -> int:
    """Get the results of a neural network for a list of boards

    Args:
        nn (NeuralNetwork): neural network to use
        inputs (List[Board]): boards to compute

    Returns:
        int: error code
    """
    for data in inputs:
        board = np.array(data.boards)
        res = nn.forward(board)
        interpret_result(res, data.turn)
    return 0
