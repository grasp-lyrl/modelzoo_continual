import os
import yaml


def fetch_configs(fname):
    # Get config
    with open(os.path.join(fname), 'r') as myfile:
        data = yaml.load(myfile, Loader=yaml.FullLoader)
    return data
