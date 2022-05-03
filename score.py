import os
import warnings
import sys

import pandas as pd
import numpy as np

import mlflow
import mlflow.sklearn
import pickle

def load_test_data(data_path):
    data = pd.read_csv(data_path)

    # The predicted column is "median_house_value"
    test_x = data.drop(["median_house_value"], axis=1)
    test_y = data[["median_house_value"]]
    return test_x, test_y
    

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    data_path = "housing_test.csv"
    filename = "finalized_model.sav" 
    train_x, train_y, test_x, test_y = load_test_data(data_path)
    loaded_model = pickle.load(open(filename, 'rb'))
    result = loaded_model.score(X_test, Y_test)
    print(result)

    
	