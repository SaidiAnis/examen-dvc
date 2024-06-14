from sklearn.ensemble import GradientBoostingRegressor
import pandas as pd
import pickle

def training():
    print("Training Gradient Boosting Regressor with Best Parameters")

    # Load scaled features and labels
    X_train = pd.read_csv("data/processed_data/X_train.csv", index_col = "date")
    y_train = pd.read_csv("data/processed_data/y_train.csv", index_col = "date")

    y_train = y_train.values.ravel()

    # Load the best parameters obtained from the grid search
    with open("models/best_params.pkl", "rb") as f:
        best_params = pickle.load(f)

    # Initialize the Gradient Boosting Regressor with the best parameters
    model = GradientBoostingRegressor(**best_params)
    model.fit(X_train, y_train)

    # Save the trained model
    with open("models/gbr_model.pkl", "wb") as f:
        pickle.dump(model, f)

if __name__ == '__main__':
    training()


