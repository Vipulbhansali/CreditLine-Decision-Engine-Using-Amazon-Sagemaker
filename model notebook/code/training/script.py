# | filename: script.py
# | code-line-numbers: true

import argparse
import json
import os
import tarfile

from pathlib import Path
from comet_ml import Experiment

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score


def train_xgboost(
    model_directory,
    train_path,
    validation_path,
    pipeline_path,
    experiment,
    num_boost_round=50  # Default value set to 50
):
    X_train = pd.read_csv(Path(train_path) / "train.csv")
    y_train = X_train[X_train.columns[-1]]
    X_train = X_train.drop(X_train.columns[-1], axis=1)

    X_validation = pd.read_csv(Path(validation_path) / "validation.csv")
    y_validation = X_validation[X_validation.columns[-1]]
    X_validation = X_validation.drop(X_validation.columns[-1], axis=1)

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_validation, label=y_validation)

    params = {
        "objective": "binary:logistic",
        "min_child_weight": 1,
        "eval_metric": "logloss",
        "learning_rate": 0.1,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "alpha": 0.1,
        "lambda": 1,
        "gamma": 0.1,
    }

    evals = [(dtrain, "train"), (dvalid, "valid")]

    bst = xgb.train(params, dtrain, num_boost_round=num_boost_round, evals=evals, verbose_eval=2)

    pred_probs = bst.predict(dvalid)
    predictions = (pred_probs > 0.5).astype(int)
    val_accuracy = accuracy_score(y_validation, predictions)
    print(f"Validation accuracy: {val_accuracy}")

    model_filepath = Path(model_directory) / "xgboost_model.json"
    bst.save_model(model_filepath)

    # Create a tar.gz file and add the model file with the correct name
    with tarfile.open(Path(model_directory) / "model.tar.gz", "w:gz") as tar:
        tar.add(model_filepath, arcname="xgboost_model.json")

    # Extract existing files from model.tar.gz (if needed)
    with tarfile.open(Path(pipeline_path) / "model.tar.gz", "r:gz") as tar:
        tar.extractall(model_directory)

    if experiment:
        experiment.log_parameters(
            {
                "num_boost_round": num_boost_round,
            }
        )
        experiment.log_dataset_hash(X_train)
        experiment.log_confusion_matrix(
            y_validation.astype(int), predictions.astype(int)
        )
        experiment.log_model("creditline_xgboost", model_filepath.as_posix())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args, _ = parser.parse_known_args()

    comet_api_key = os.environ.get("COMET_API_KEY", None)
    comet_project_name = os.environ.get("COMET_PROJECT_NAME", None)

    experiment = (
        Experiment(
            project_name=comet_project_name,
            api_key=comet_api_key,
            auto_metric_logging=True,
            auto_param_logging=True,
            log_code=True,
        )
        if comet_api_key and comet_project_name
        else None
    )

    training_env = json.loads(os.environ.get("SM_TRAINING_ENV", {}))
    job_name = training_env.get("job_name", None) if training_env else None

    if job_name and experiment:
        experiment.set_name(job_name)

    train_xgboost(
        model_directory=os.environ["SM_MODEL_DIR"],
        train_path=os.environ["SM_CHANNEL_TRAIN"],
        validation_path=os.environ["SM_CHANNEL_VALIDATION"],
        pipeline_path=os.environ["SM_CHANNEL_PIPELINE"],
        experiment=experiment
    )
