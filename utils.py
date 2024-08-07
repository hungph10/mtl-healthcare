import os
import random
import torch

import numpy as np
import pandas as pd


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def recover_df(array):
    array = array.reshape(-1, array.shape[-1])
    df = pd.DataFrame(data=array, columns=["x", 'y', 'z', 'posture', 'breath_rate'])
    return df



def plot_dict_3d(data, color):
    """
    Plots a 3D scatter plot using a dictionary with keys 'x', 'y', and 'z'.

    Parameters:
    data (dict): Dictionary with keys 'x', 'y', and 'z' each containing a list of values.
    """
    # Set the modern theme
    plt.style.use('seaborn')

    color_map = {
        0: "blue",
        1: "green",
        2: "red",
        3: "purple",
        4: "orange",
        5: "teal",
        6: "fuchsia",
        7: "gray",
        8: "olive",
        9: "maroon",
        10: "navy",
        11: "lime"
    }

    color = [color_map[int(c)] for c in color]
    # Create a figure and a 3D axis
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Extract the x, y, and z data
    x = data['x']
    y = data['y']
    z = data['z']

    # Plot the data
    ax.scatter(x, y, z, c=color, marker='o')

    # Add a title and labels
    ax.set_title('3D Scatter Plot with Modern Theme')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Show the plot
    plt.show()




def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


def resolution(data: np.array, param: int):
    # Reshape array A into a  X param X len_after_resolution matrix
    len_after = data.shape[0] // param
    try:
        data_after = data[:len_after*param]
        data_after = data_after.reshape(len_after, param, data.shape[1])
        return data_after.mean(axis=1)
    except:
        print("Length data isn't divisible for param!")
        raise 


def preprocess_data(
    df: pd.DataFrame,
    len_point: int,
    resolution_param: int,
    col_breath_rate,
    col_id="id"
):
    len_data = len(df)
    num_data = len_data // len_point
    new_df = df.drop(columns=[col_breath_rate])
    af_dataset = np.array([
        resolution(
            data=new_df.iloc[i*len_point : (i+1)*len_point].values,
            param=resolution_param
        )
        for i in range(num_data)
    ])
    af_labels = np.array([
        df[col_breath_rate][(i+1) * len_point - 1]
        for i in range(num_data)
    ])
    return af_dataset, af_labels



import json

def pretty_print_json(data):
    """
    Print JSON-like data with automatic indentation.

    Parameters:
    - data: JSON-like data (e.g., a dictionary or a list)
    """
    pretty_json = json.dumps(data, indent=4)
    print(pretty_json, flush=True)


import json

def save_json(data, file_path):
    """
    Save JSON-like data to a file.

    Parameters:
    - data: JSON-like data (e.g., a dictionary or a list)
    - file_path: Path to the file where JSON data will be saved
    """
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

