import pandas as pd
import numpy as np
import os
import json
import shutil
import json
import argparse
from xml.dom import minidom
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import sklearn.metrics as skmet
from sklearn.ensemble import RandomForestClassifier
from joblib import dump, load


def load_config(config_path):
    """Load json with user params """
    with open(config_path) as json_file:
        data = json.load(json_file)
    return data
    
    
def load_trackmate_path(trackmate_path):
    """A function to load FILTERED spots """
    return pd.read_csv(trackmate_path)

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-config", nargs='?', default='modelTrain/config.json', 
                        help="The path to a processed alignment table.")
    args = parser.parse_args()
    
    # argument parsing 
    config_path = args.config
    config = load_config(config_path)
    # load all config params as variables
    # these are all UPPERCASE
    locals().update(config)
    
    # load the filtered trackmate file
    df = load_trackmate_path(TRACKMATE_PATH)
    
  
    