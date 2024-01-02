import numpy as np
import pandas as pd


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
    new_df = df.drop(columns=[col_id, col_breath_rate])
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

