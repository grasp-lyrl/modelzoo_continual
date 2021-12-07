import os
import yaml


def fetch_configs(fname):
    """
    Load yaml config file
    """
    with open(os.path.join(fname), 'r') as myfile:
        data = yaml.load(myfile, Loader=yaml.SafeLoader)
    return data
