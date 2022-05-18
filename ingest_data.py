import os
import warnings
import sys

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet

import mlflow
import mlflow.sklearn


def load_data(data_path):
    data = pd.read_csv(data_path)

    # Split the data into training and test sets. (0.75, 0.25) split.
    train, test = train_test_split(data)

    # The predicted column is "median_house_value"
    train_x = train.drop(["median_house_value"], axis=1)
    test_x = test.drop(["median_house_value"], axis=1)
    train_y = train[["median_house_value"]]
    test_y = test[["median_house_value"]]
    return train_x, train_y, test_x, test_y


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    data_path = "housing.csv"
    train_x, train_y, test_x, test_y = load_data(data_path)