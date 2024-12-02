#!/usr/bin/env python3
"""main for generator"""
import typer
from typing import List

def generate(config: str) -> None:
    print("heello world")
    return

def main_generator(configs: List[str] = typer.Option(..., '', '', help="config file")):
    """
        generator
    """
    for conf in configs:
        generate("toto")
#
def main_typer():
    """main to call the cli handling
    """
    try:
        typer.run(main_generator)
    except Exception as err:
        print(err.args[0])
        exit(84)
#
#
if __name__ == "__main__":
    typer.run(main_typer)
