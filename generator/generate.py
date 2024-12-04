"""generate a neural network
"""
from sys import stderr

class Conf_parameters:
    """parse and contains a configuration
    """

    def __init__(self, filecontent : str):
        self.filecontent : str = filecontent

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

        self.conf_file = None
        self.conf : Conf_parameters = None
        try:
            self.conf_file = open(file, 'r', encoding='utf-8')
            self.conf = Conf_parameters(self.conf_file.read())
        except Exception as err:
            print(f"Opening of {file} failed", file=stderr)
            if len(err.args) > 1:
                print('->', err.args[1])
            raise Exception
        self.conf_file.close()
        print(f"{nb} neural networks will be generated with this configuration")
        self.nb : int = nb
        self.nn_names : list = []
        print("Configuration properly loaded\n")

    def getNames(self) -> list[str]:
        """get the names of the files containing the resulting neural networks

        Returns:
            list[str]: list of file names
        """
        names : list[str] = []
        for i in range(self.nb):
            names.append(f"{self.name}_{i}.nn")
        return names


def generate_nn(filename: str, conf: Conf_parameters) -> None:
    """Generate a neural network and put it in the file

    Args:
        filename (str): name of the resulting file
    """
    print(f"Generating {filename}")

    file = None
    try:
        file = open(filename, 'a', encoding='utf-8')
    except Exception as err:
        print(f"Opening of {filename} failed", file=stderr)
        if len(err.args) > 1:
            print('->', err.args[1])
        raise RuntimeError


    # TODO put code here

    file.write(conf.filecontent) # TODO write the file
    file.close()

    return



def generate_files(conf: Config) -> None:
    """generate neurals network from a config

    Args:
        conf (Config): configuration to use
    """
    names = conf.getNames()
    print(f"The files {names} will be generated")
    for file in names:
        generate_nn(file, conf.conf)
    return
