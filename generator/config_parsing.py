"""class to parse the config file
"""
from sys import stderr
from typing import List
import json


class Conf_parameters:
    """parse and contains a configuration
    """

    def __parse_epoch(self, content: dict):
        if not "epoch" in content.keys():
            raise KeyError("'epoch' must be set in the config")
        epoch = content["epoch"]
        if not isinstance(epoch, int) or epoch <= 0:
            raise ValueError("'epoch' must be a positive int")
        self.epoch = epoch

    def __parse_eta(self, content: dict):
        if not "eta" in content.keys():
            raise KeyError("'eta' must be set in the config")
        eta = content["eta"]
        if not isinstance(eta, (float, int)) or eta <= 0:
            raise ValueError("'eta' must be a positive float")
        self.eta = eta

    def __parse_fully_connected(self, content: dict):
        if not "fully_connected" in content.keys():
            raise KeyError("'fully_connected' must be set in the config")
        fully = content["fully_connected"]
        if not isinstance(fully, list) or any(not isinstance(x, int) or x <= 0 for x in fully):
            raise ValueError(
                "'fully_connected' must be a list of positive int")
        self.fully_connected = fully

    def __is_valid_conv_layer(self, layer) -> bool:
        expected_keys = {
            "num_filters": lambda x: isinstance(x, int) and x > 0,
            "input_depth": lambda x: isinstance(x, int) and x > 0,
            "kernel_size": lambda x: isinstance(x, int) and x > 0,
            "eta": lambda x: isinstance(x, (float, int)) and x > 0
        }
        if not isinstance(layer, dict):
            print("layer must be a dict", file=stderr)
            return False
        for key, condition in expected_keys.items():
            if key not in layer or not condition(layer[key]):
                return False
        return True

    def __parse_conv_layers(self, content: dict):
        if not "conv_layers" in content.keys():
            raise KeyError("'conv_layers' must be set in the config")
        conv = content["conv_layers"]
        if not isinstance(conv, list) or any(
                not self.__is_valid_conv_layer(layer) for layer in conv):
            raise ValueError(
                "'conv_layers' must respect the good format")
        self.conv_layers = conv

    def __init__(self, file: str):
        self.epoch: int = 0
        self.eta: float = 0
        self.fully_connected: List[int] = []
        self.conv_layers: List[dict] = []
        content = {}
        try:
            with open(file, 'r', encoding='utf-8') as file:
                content = json.load(file)
        except Exception as exc:
            raise RuntimeError(exc.args[0]) from exc

        for key in content.keys():
            if key not in ["conv_layers", "fully_connected", "eta", "epoch"]:
                raise KeyError(
                    f"Key '{key}' isn't expected in the config file")

        self.__parse_epoch(content)
        self.__parse_eta(content)
        self.__parse_fully_connected(content)
        self.__parse_conv_layers(content)


class Config:
    """contains the config and number for a neural network, is used for basis to generate the network
    """

    def __init__(self, file: str, nb: int):
        print(f"Creating config for : {file}")
        self.name = file

        if file[-5::] != '.conf':
            print("file name should end with .conf", file=stderr)
        else:
            self.name = file[:-5]

        self.conf: Conf_parameters = None
        try:
            self.conf = Conf_parameters(file)
        except (RuntimeError, ValueError) as err:
            print(f"Parsing of {file} failed\n{err.args}", file=stderr)
            raise RuntimeError from err
        self.nb: int = nb
        self.nn_names: list = []
        print("Configuration properly loaded")
        print(f"{nb} neural networks will be generated with this configuration\n")

    def get_names(self) -> list[str]:
        """get the names of the files containing the resulting neural networks

        Returns:
            list[str]: list of file names
        """
        names: list[str] = []
        for i in range(self.nb):
            names.append(f"{self.name}_{i}.nn")
        return names
