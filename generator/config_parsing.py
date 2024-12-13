"""class to parse the config file
"""
from typing import List


class Conf_parameters:
    """parse and contains a configuration
    """

    def __parse_layers(self, line: str):
        self.layers = eval(line)

    def __parse_epoch(self, line: str):
        self.epoch = int(line)

    def __parse_eta(self, line: str):
        self.eta = float(line)

    def __init__(self, file: str):
        self.layers: List[int] = []
        self.epoch: int = 0
        self.eta: float = 0
        content = ""
        fd = None
        try:
            fd = open(file, 'r', encoding='utf-8')
            content = fd.read()
            fd.close()
        except Exception as exc:
            raise RuntimeError(f"Config file '{file}' failed to open") from exc
        parse_funcs: dict = {"layers": self.__parse_layers,
                             "epoch": self.__parse_epoch, "eta": self.__parse_eta}
        print(content.split('\n'))
        for i, line in enumerate(content.split('\n')):
            line.strip()
            if line == '' or line[0] == '#':
                continue
            elts = line.split('=')
            if len(elts) != 2 or not elts[0].strip() in parse_funcs.keys():
                raise ValueError(f"Error in {file} at line {i}:\n\t{line}")
            try:
                parse_funcs[elts[0].strip()](elts[1].strip())
            except Exception as err:
                raise ValueError(f"Error in {file}:\n\t{i}:{line}") from err
        print(self.layers, self.epoch, self.eta)


class Config:
    """contains the config and number for a neural network, is used for basis to generate the network
    """

    def __init__(self, file: str, nb: int):
        print(f"Creating config for : {file}")
        self.name = file

        if file[-5::] != '.conf':
            print("file name should end with .conf", file=stderr)
        else:
            self.name = file[-4::]

        print()

        self.conf: Conf_parameters = None
        try:
            self.conf = Conf_parameters(file)
        except (RuntimeError, ValueError) as err:
            print(f"Parsing of {file} failed\n{err.args}", file=stderr)
            raise RuntimeError from err
        print(f"{nb} neural networks will be generated with this configuration")
        self.nb: int = nb
        self.nn_names: list = []
        print("Configuration properly loaded\n")

    def get_names(self) -> list[str]:
        """get the names of the files containing the resulting neural networks

        Returns:
            list[str]: list of file names
        """
        names: list[str] = []
        for i in range(self.nb):
            names.append(f"{self.name}_{i}.nn")
        return names
