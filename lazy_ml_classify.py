from sklearn.model_selection import train_test_split
import argparse
import numpy as np

from train_utils import get_dataset
from lazypredict import LazyClassifier


def parse_arguments():
    parser = argparse.ArgumentParser(description="Hyper parameters for training")
    parser.add_argument('--data_path', type=str, default=None, help='Path to the npz data training')
    parser.add_argument('--output_report_csv', type=str, default=None, help='Path to the npz data training')

    args = parser.parse_args()
    return args


def dataset_to_matrix(dataset):
    x, y = dataset[:]
    x = np.concatenate(x.numpy(), axis=0)
    y = np.concatenate(y.numpy(), axis=0)
    return x, y


if __name__ == "__main__":
    args = parse_arguments()

    train_dataset, dev_dataset, test_dataset = get_dataset(args.data_path, task="Classify")

    x_train, y_train = dataset_to_matrix(train_dataset)
    x_dev, y_dev = dataset_to_matrix(dev_dataset)
    x_test, y_test = dataset_to_matrix(test_dataset)   


    ## Sampling data train and test
    # x_train, _, y_train, _ = train_test_split(x_train, y_train, test_size=0.99)

    # x_test, _, y_test, _ = train_test_split(x_test, y_test, test_size=0.9)

    # print(x_train.shape, x_test.shape)

    clf = LazyClassifier(verbose=0,ignore_warnings=True, custom_metric=None)
    models, predictions = clf.fit(x_train, x_test, y_train, y_test)

    predictions.to_csv(args.output_report_csv)