#!/usr/bin/env python3
"""main for generator"""
from sys import stderr
from typing import Optional, List
import typer
from typing_extensions import Annotated
from generator.generate import Config, generate_files


def main_generator(config_file_1 : Annotated [str, typer.Argument(help="Configuration file containing description of a neural network we want to generate.")],
                   nb_1 : Annotated [int, typer.Argument(help="Number of neural networks to generate based on the configuration file.")],
                   other_configs: Annotated [Optional[List[str]], typer.Argument(help="Other confings files and there numbers.")] = None):
    """
        generator
    """
    configs : List[Config] = []
    try:
        new: Config = Config(config_file_1, nb_1)
        configs.append(new)
        if len(other_configs) % 2 != 0:
            raise IndexError("Invlaid number of arguments, each config must have a number")
        for i in range(0, len(other_configs), 2):
            try:
                configs.append(Config(other_configs[i], int(other_configs[i + 1])))
            except ValueError:
                raise ValueError(f"Error: Arg :'{other_configs[i]}' should be a valid int")
    except Exception as err:
        if len(err.args) > 0:
            print(err.args[0], file=stderr)
        raise typer.Exit(84)
    for conf in configs:
        generate_files(conf)

if __name__ == "__main__":
    typer.run(main_generator)
