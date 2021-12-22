import pandas as pd
import numpy as np
import os
from tqdm.notebook import tqdm
import shutil
from importlib import reload
import skimage.io
import matplotlib 
import matplotlib.pyplot as plt 
import matplotlib.patches as patches
import json
import argparse
import subprocess


def load_config(config_path):
    """Load json with user params """
    with open(config_path) as json_file:
        data = json.load(json_file)
    return data

def format_fiji_args():
    fiji_args = []
    for k, v in FIJI_ARGS.items():
        fiji_args.append(f'{k}="{v}"')
    
    fiji_args = ",".join(fiji_args)
    return fiji_args

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-config", nargs='?', default='trackMe/config.json', 
                        help="The path to a processed alignment table.")
    args = parser.parse_args()
    
    # argument parsing 
    config_path = args.config
    config = load_config(config_path)
    # load all config params as variables
    # these are all UPPERCASE
    locals().update(config)
    
    fiji_args = str(format_fiji_args())
    bashCommand = ["xvfb-run", "-a", f"{FIJI}", "--ij2", "--console", "--run", f'{SCRIPT}', fiji_args]
    runner = subprocess.run(bashCommand, check=True, text=True, shell=False)
#     print(runner)
