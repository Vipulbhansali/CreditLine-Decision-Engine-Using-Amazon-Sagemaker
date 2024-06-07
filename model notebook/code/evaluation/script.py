# | filename: script.py
# | code-line-numbers: true

import json
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score
from tensorflow import keras
import xgboost as xgb


def evaluate(model_path, test_path, output_path):
    X_test = pd.read_csv(Path(test_path) / "test.csv")
    y_test = X_test[X_test.columns[-1]]
    X_test = X_test.drop(X_test.columns[-1], axis=1)

    # Let's now extract the model package so we can load
    # it in memory.
    with tarfile.open(Path(model_path) / "model.tar.gz") as tar:
        tar.extractall(path=Path(model_path))

    bst = xgb.Booster()
    bst.load_model(Path(model_path) / "xgboost_model.json")

    dtest = xgb.DMatrix(X_test, label=y_test)
    pred_probs = bst.predict(dtest)


    predictions = (pred_probs > 0.5).astype(int)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)


    print(f" Accuracy: {accuracy}")
    print(f" precision: {precision}")
    print(f" recall: {recall}")

    # Let's create an evaluation report using the model accuracy.
    evaluation_report = {
    "metrics": {
        "accuracy": {"value": accuracy},
        "precision": {"value": precision},
        "recall": {"value": recall},
    },
}

    Path(output_path).mkdir(parents=True, exist_ok=True)
    with open(Path(output_path) / "evaluation.json", "w") as f:
        f.write(json.dumps(evaluation_report))


if __name__ == "__main__":
    evaluate(
        model_path="/opt/ml/processing/model/",
        test_path="/opt/ml/processing/test/",
        output_path="/opt/ml/processing/evaluation/",
    )
