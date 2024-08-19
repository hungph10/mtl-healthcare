import pandas as pd

from utils import preprocess_data, resolution, recover_df
from augment import *


inp_path = "data/split_val_from_raw_1508/split_val_from_train.npz"
data = np.load(inp_path)

out_path = "data/split_val_from_raw_1508/train_val_test-1508.npz"
# data = np.load("data/update-0208/train_dev_test_split.npz")

# train_path = "data/cls_regr_16_person/train.csv"
# test_path = "data/cls_regr_16_person/test.csv"
# train_df = pd.read_csv(train_path, index_col=False)
# test_df = pd.read_csv(test_path, index_col=False)

train_df = recover_df(data['train'])
dev_df = recover_df(data['dev'])
test_df = recover_df(data['test'])


len_point = 3000
resolution_param = 10
col_breath_rate = "breath_rate"

X_train, Y_train = preprocess_data(
    df=train_df,
    len_point=len_point,
    resolution_param=resolution_param,
    col_breath_rate=col_breath_rate
)

X_dev, Y_dev = preprocess_data(
    df=dev_df,
    len_point=len_point,
    resolution_param=resolution_param,
    col_breath_rate=col_breath_rate
)

X_test, Y_test = preprocess_data(
    df=test_df,
    len_point=len_point,
    resolution_param=resolution_param,
    col_breath_rate=col_breath_rate
)

from augment import Augment

augmenter = Augment(features=X_train, labels=Y_train)
x_train, y_train = augmenter(augment_times=3)


import numpy as np
data = {
    "x_train": x_train,
    "y_train": y_train,
    "x_dev": X_dev,
    "y_dev": Y_dev,
    "x_test": X_test,
    "y_test": Y_test
}

np.savez(out_path, **data)