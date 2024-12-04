"""generate a neural network
"""
from sys import stderr

class Config:
    """contains the config and number for a neural network, is used for basis to generate the network
    """
    def __init__(self, file : str, nb: int):
        print(f"Creating config for : {file}")
        self.name = file

        if file[-5::] != '.conf':
            print("file name should end with .conf", file=stderr)
        else:
            self.name = file[-4::]

        print()

        self.conf = None
        try:
            self.conf = open(file, 'r', encoding='utf-8')
            self.content = self.conf.read()
        except Exception as err:
            print(f"Opening of {file} failed", file=stderr)
            if len(err.args) > 1:
                print('->', err.args[1])
            raise Exception

        print(f"{nb} neural networks will be generated with this configuration")
        self.nb : int = nb
        self.nn_names : list = []
        print("Configuration properly loaded\n")



def generate(conf: Config) -> None:
    """generate neurals network from a config

    Args:
        conf (Config): configuration to use
    """
    return
