from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
import pandas as pd
import pickle
import yaml

def grid_search():
    print("Performing Grid Search for Gradient Boosting Regressor")

    # Load parameters from YAML file
    params = yaml.safe_load(open("src/params.yaml"))["grid_search"]
    
    # Load scaled features and labels
    X_train = pd.read_csv("data/processed_data/X_train.csv", index_col = "date")
    y_train = pd.read_csv("data/processed_data/y_train.csv", index_col = "date")

    y_train = y_train.values.ravel()

    # Define the parameter grid
    param_grid = {
        'n_estimators': params["n_est"],
        'max_depth': params["m_depth"],
        'learning_rate': params["lr"],
        'min_samples_split': params["min_split"],
        'min_samples_leaf': params["min_leaf"]
    }

    # Initialize the Gradient Boosting Regressor
    model = GradientBoostingRegressor()

    # Perform Grid Search
    grid_search = GridSearchCV(model, param_grid, cv=params["cv"], scoring=params["scoring"])
    grid_search.fit(X_train, y_train)

    # Print the best parameters
    best_params = grid_search.best_params_
    print("Best Parameters:", best_params)

    # Save the best parameters to a file
    with open("models/best_params.pkl", "wb") as f:
        pickle.dump(best_params, f)

if __name__ == '__main__':
    grid_search()


