"""
Split raw data into training and testing data
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import yaml

# Reading the parameters to use for the split
params = yaml.safe_load(open("src/params.yaml"))["data-split"]
split = params["split"]
seed = params["seed"]

def data_split():
    # Load the data
    print("Loading data from given folder")
    df = pd.read_csv('data/raw_data/raw.csv', index_col="date")
    print("done")

    # Separating into target variable and features
    X = df.drop("silica_concentrate", axis = 1)
    y = df["silica_concentrate"]

    # Splitting the data
    print("Splitting data into train and test")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split, random_state=seed)
    print("done")

    # Saving the data into the processed file
    X_train.to_csv('data/processed_data/X_train.csv', index=True)
    X_test.to_csv('data/processed_data/X_test.csv', index=True)
    y_train.to_csv('data/processed_data/y_train.csv', index=True)
    y_test.to_csv('data/processed_data/y_test.csv', index=True)
    print("Saved data into processed folder")


if __name__ == '__main__':
    data_split()
