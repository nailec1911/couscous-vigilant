#!/usr/bin/env python3
"""main for analyzer"""
import argparse
from sys import stderr
from analyzer.predict import predict
from analyzer.train import train
def parse_arguments():
    parser = argparse.ArgumentParser(prog = "my_torch_analyzer",
        description="A neural network analyzer for training or prediction using chessboard data."
    )

    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        '--predict',
        action='store_true',
        help="Launch the neural network in prediction mode. Each chessboard in FILE" \
            "must contain inputs to send to the neural network in FEN notation, and optionally an" \
            "expected output."
    )
    mode_group.add_argument(
        '--train',
        action='store_true',
        help="Launch the neural network in training mode. Each chessboard in FILE must" \
            "contain inputs to send to the neural network in FEN notation and the expected output" \
            "separated by space. If specified, the newly trained neural network will be saved in" \
            "SAVEFILE. Otherwise, it will be saved in the original LOADFILE." \
    )

    parser.add_argument(
        '--save',
        metavar='SAVEFILE',
        type=argparse.FileType('w'),
        help="Save neural network into SAVEFILE. Only works in train mode."
    )

    parser.add_argument(
        'LOADFILE',
        metavar='LOADFILE',
        type=argparse.FileType('r'),
        help="File containing an artificial neural network"
    )
    parser.add_argument(
        'FILE',
        metavar='FILE',
        type=argparse.FileType('r'),
        help="File containing chessboards."
    )

    args = parser.parse_args()

    if args.save and not args.train:
        parser.error("--save can only be used with --train.")
    if args.train and not args.save:
        parser.error("--train requires the --save option.")

    return args


def main_analyzer(args: argparse.Namespace) -> None:
    """main for the

    Args:
        args (argparse.Namespace): parsed args
    """
    if args.predict:
        predict()
        return
    if args.train:
        train()
        return

if __name__ == "__main__":
    try:
        args = parse_arguments()
        print(type(args))
    except Exception as e:
        print(e, file=stderr)
