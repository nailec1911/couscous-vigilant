#!/usr/bin/env python3
"""main for analyzer"""
import argparse
from sys import stderr
from typing import List
from analyzer.predict import predict
from analyzer.train import train
from neural_network.neural_network import NeuralNetwork
from pickle import load
from analyzer.board_parsing import Board


def parse_arguments():
    parser = argparse.ArgumentParser(
        prog="my_torch_analyzer",
        description="A neural network analyzer for training or prediction using chessboard data.",
    )

    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--predict",
        action="store_true",
        help="Launch the neural network in prediction mode. Each chessboard in FILE"
        "must contain inputs to send to the neural network in FEN notation, and optionally an"
        "expected output.",
    )
    mode_group.add_argument(
        "--train",
        action="store_true",
        help="Launch the neural network in training mode. Each chessboard in FILE must"
        "contain inputs to send to the neural network in FEN notation and the expected output"
        "separated by space. If specified, the newly trained neural network will be saved in"
        "SAVEFILE. Otherwise, it will be saved in the original LOADFILE.",
    )

    parser.add_argument(
        "--save",
        metavar="SAVEFILE",
        type=str,
        help="Save neural network into SAVEFILE. Only works in train mode.",
    )

    parser.add_argument(
        "LOADFILE",
        metavar="LOADFILE",
        type=str,
        help="File containing an artificial neural network",
    )
    parser.add_argument(
        "FILE",
        metavar="FILE",
        type=str,
        help="File containing chessboards.",
    )

    args = parser.parse_args()

    if args.save and not args.train:
        parser.error("--save can only be used with --train.")
    if args.train and not args.save:
        args.save = args.LOADFILE

    return args


def main_analyzer(args: argparse.Namespace) -> None:
    """main for the

    Args:
        args (argparse.Namespace): parsed args
    """

    nn: NeuralNetwork = None
    try:
        with open(args.LOADFILE, 'rb') as file:
            nn = load(file)
    except Exception as err:
        raise FileNotFoundError(
            f"Neural network from file '{args.LOADFILE}' didn't load properly") from err

    inputs: List[Board] = []
    try:
        with open(args.FILE, 'r', encoding='utf-8') as file:
            lines = [e[:-1] for e in list(file)]
            for e in lines:
                inputs.append(Board(e, args.train))
    except Exception as err:
        raise FileNotFoundError(
            f"Inputs from '{args.FILE}' didn't load properly") from err

    if args.predict:
        return predict(nn, inputs)
    if args.train:
        return train(nn, args.save, inputs)
    return 84  # should not append


if __name__ == "__main__":
    try:
        arg = parse_arguments()
        main_analyzer(arg)
    except Exception as e:
        print(e, file=stderr)
        exit(84)
