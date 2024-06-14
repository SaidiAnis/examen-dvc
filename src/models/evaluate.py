import sklearn.metrics as metrics
import pickle
import json
import numpy as np
import pandas as pd

def evaluate():
    print("Model Evaluation")
    X_test = pd.read_csv("data/processed_data/X_test_scaled.csv", index_col = "date")
    y_test = pd.read_csv("data/processed_data/y_test.csv", index_col = "date")

    y_test = y_test.values.ravel()

    model = pickle.load(open("models/gbr_model.pkl", "rb"))
    predictions = model.predict(X_test)

    prediction_csv = pd.DataFrame({"target_labels": y_test,
                                   "predicted_labels": predictions})
    prediction_csv.to_csv("data/prediction.csv", index=False)

    mse = metrics.mean_squared_error(y_test, predictions)
    r2 = metrics.r2_score(y_test, predictions)

    with open("metrics/scores.json", "w") as fd:
        json.dump({"mse": mse, "r2": r2}, fd, indent=4)


if __name__ == '__main__':
    evaluate()
