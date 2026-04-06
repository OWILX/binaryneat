import numpy as np
import configparser as cp
import os


# logic gates
gates = {
    "AND":  lambda x, y: x & y,
    "NAND": lambda x, y: ~(x & y),
    "OR":   lambda x, y: x | y,
    "NOR":  lambda x, y: ~(x | y),
    "XOR":  lambda x, y: x ^ y,
    "XNOR": lambda x, y: ~(x ^ y),
    "NOT":  lambda x, y: ~x,
}

GATE_KEYS = list(gates.keys())

def get_config():
    config = cp.ConfigParser()

    # get the directory where this file is located
    dirpath = os.path.abspath(os.path.dirname(__file__))
    default_config_path = os.path.join(dirpath, 'default_config.ini')
    config.read(default_config_path)

    # load users config if it exists
    execution_path = os.path.abspath(os.getcwd())
    user_config_path = os.path.join(execution_path, 'config.ini')
    if os.path.exists(user_config_path):
        config.read(user_config_path)

    return config