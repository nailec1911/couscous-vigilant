"""generate a neural network
"""
from sys import stderr
from generator.config_parsing import Config, Conf_parameters


def generate_nn(filename: str, conf: Conf_parameters) -> None:
    """Generate a neural network and put it in the file

    Args:
        filename (str): name of the resulting file
    """
    print(f"Generating {filename}")

    file = None
    try:
        file = open(filename, 'w', encoding='utf-8')
    except Exception as err:
        print(f"Opening of {filename} failed", file=stderr)
        if len(err.args) > 1:
            print('->', err.args[1])
        raise RuntimeError

    # TODO put code here

    file.write(",dsmlfsdmf")  # TODO write the file
    file.close()

    return


def generate_files(conf: Config) -> None:
    """generate neurals network from a config

    Args:
        conf (Config): configuration to use
    """
    names = conf.get_names()
    print(f"The files {names} will be generated")
    for file in names:
        generate_nn(file, conf.conf)
    return
