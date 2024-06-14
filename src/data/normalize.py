"""
Standard Scaling the raw data
"""
from sklearn.preprocessing import StandardScaler
import pandas as pd

def normalize():
    print("Normalizing the data")

    print("Loading split data")
    X_train = pd.read_csv("./data/processed_data/X_train.csv", index_col="date")
    X_test = pd.read_csv("./data/processed_data/X_test.csv", index_col="date")
    print("done")

    print("Scaling data with Standard Scaler")
    scaling = StandardScaler()
    scaling.fit(X_train)
    
    X_train_scaled = pd.DataFrame(scaling.transform(X_train), index=X_train.index, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaling.transform(X_test), index=X_test.index, columns=X_test.columns)
    print("done")

    # Save the scaled data back to the files
    X_train_scaled.to_csv("./data/processed_data/X_train_scaled.csv", index = True)
    X_test_scaled.to_csv("./data/processed_data/X_test_scaled.csv", index = True)

if __name__ == '__main__':
    normalize()