"""
@author:    Ge Li, ge.li@kit.edu
@brief:     Utilities
"""

# Import Python libs
import csv
import json
import os
import random

import numpy as np
import pandas as pd
from mnist import MNIST
from natsort import os_sorted
import nmp.util as util





def read_dataset(dataset_name: str,
                 shuffle: bool = False,
                 seed=None) -> (list, list):
    """
    Read raw data from files

    Args:
        dataset_name: name of dataset to be read
        shuffle: shuffle the order of dataset files when reading
        seed: random seed

    Returns:
        list_pd_df: a list of pandas DataFrames with time-dependent data
        list_pd_df_static: ... time-dependent data, can be None

    """
    # Get dir to dataset
    dataset_dir = util.get_dataset_dir(dataset_name)

    # Get all data-file names
    file_names = util.get_file_names_in_directory(dataset_dir)

    # Check file names for both time-dependent and time-independent data exist
    num_files = len(file_names)
    file_names = os_sorted(file_names)

    # Check if both time-dependent and time-independent dataset exist
    if all(['static' in name for name in file_names]):
        # Only time-independent dataset
        list_pd_df = [pd.DataFrame() for data_file in file_names]
        # Construct a empty dataset for time independent data
        list_pd_df_static = [pd.read_csv(os.path.join(dataset_dir, data_file),
                                         quoting=csv.QUOTE_ALL)
                             for data_file in file_names]
    elif all(['static' not in name for name in file_names]):
        # Only time-dependent dataset
        list_pd_df = [pd.read_csv(os.path.join(dataset_dir, data_file),
                                  quoting=csv.QUOTE_ALL)
                      for data_file in file_names]
        # Construct a empty dataset for time independent data
        list_pd_df_static = [pd.DataFrame() for data_file in file_names]
    else:
        # Both exist
        assert \
            all(['static' not in name for name in file_names[:num_files // 2]])
        assert all(['static' in name for name in file_names[num_files // 2:]])

        # Read data from files and generate list of pandas DataFrame
        list_pd_df = [pd.read_csv(os.path.join(dataset_dir, data_file),
                                  quoting=csv.QUOTE_ALL)
                      for data_file in file_names[:num_files // 2]]
        list_pd_df_static = [pd.read_csv(os.path.join(dataset_dir, data_file),
                                         quoting=csv.QUOTE_ALL)
                             for data_file in file_names[num_files // 2:]]

    if shuffle:
        list_zip = list(zip(list_pd_df, list_pd_df_static))
        random.seed(seed)
        random.shuffle(list_zip)
        list_pd_df, list_pd_df_static = zip(*list_zip)

    # Return
    return list_pd_df, list_pd_df_static
