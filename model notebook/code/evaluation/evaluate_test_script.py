
#| code-fold: true

import os
import shutil
import tarfile
import pytest
import tempfile
import pandas as pd
from pathlib import Path

from evaluation.script import evaluate
from processing.script import preprocess
from training.script import train_xgboost

# Define the path to the actual data file
DATA_FILEPATH = DATA_FILEPATH = os.getenv('LOCAL_CSV_PATH',None)

@pytest.fixture(scope="function", autouse=False)
def directory():
    directory = tempfile.mkdtemp()
    input_directory = Path(directory) / "input"
    input_directory.mkdir(parents=True, exist_ok=True)

    # Read the data, truncate to 1000 rows, and save to input directory
    df = pd.read_csv(DATA_FILEPATH)
    df.head(1000).to_csv(input_directory / "data.csv", index=False)

    directory = Path(directory)

    preprocess(base_directory=directory)

    train_xgboost(
        model_directory=directory / "model",
        train_path=directory / "train",
        validation_path=directory / "validation",
        pipeline_path=directory / "model",
        experiment=None,
        num_boost_round=25,
    )

    # After training a model, we need to prepare a package just like
    # SageMaker would. This package is what the evaluation script is
    # expecting as an input.
    with tarfile.open(directory / "model.tar.gz", "w:gz") as tar:
        tar.add(directory / "model" / "xgboost_model.json", arcname="xgboost_model.json")

    evaluate(
        model_path=directory,
        test_path=directory / "test",
        output_path=directory / "evaluation",
    )

    yield directory / "evaluation"

    shutil.rmtree(directory)


def test_evaluate_generates_evaluation_report(directory):
    output = os.listdir(directory)
    assert "evaluation.json" in output


def test_evaluation_report_contains_metrics(directory):
    with open(directory / "evaluation.json", "r") as file:
        report = json.load(file)

    assert "metrics" in report
    assert "accuracy" in report["metrics"]
    assert "precision" in report["metrics"]
    assert "recall" in report["metrics"]
