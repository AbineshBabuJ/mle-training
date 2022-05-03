import os
import warnings
import sys

import pandas as pd
import numpy as np

import mlflow
import mlflow.sklearn
import pickle
from sklearn.linear_model import LinearRegression

def load_train_data(data_path):
    data = pd.read_csv(data_path)

    # The predicted column is "median_house_value"
    train_x = data.drop(["median_house_value"], axis=1)
    train_y = data[["median_house_value"]]
    return train_x, train_y


def train(model,train_x,train_y):
    model.fit(train_x,train_y)
    filename = 'finalized_model.sav'
    pickle.dump(model, open(filename, 'wb'))
    

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    data_path = "housing_training.csv"
    train_x, train_y = load_train_data(data_path)
    model = LinearRegression()
    train(model,train_x,train_y)

    
	