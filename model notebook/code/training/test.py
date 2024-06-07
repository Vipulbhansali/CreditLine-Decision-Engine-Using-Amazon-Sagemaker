%%script false

%%ipytest -s
#| code-fold: true

import os
import shutil
import tempfile
from pathlib import Path
import pandas as pd
from processing.script import preprocess
from training.script import train_xgboost

DATA_FILEPATH = r"C:\Users\Vipul\CreditLine-Decision-Engine-Using-Amazon-Sagemaker\credit.csv"

# Run the fixture code manually
directory = tempfile.mkdtemp()
input_directory = Path(directory) / "input"
input_directory.mkdir(parents=True, exist_ok=True)
shutil.copy2(DATA_FILEPATH, input_directory / "data.csv")

# Load only 100 rows for testing
data = pd.read_csv(input_directory / "data.csv", nrows=1000).sample(n=100, random_state=42)
data.to_csv(input_directory / "data.csv", index=False)

print(f"Temporary directory created at: {directory}")

directory = Path(directory)

preprocess(base_directory=directory)

train_xgboost(
    model_directory=directory / "model",
    train_path=directory / "train", 
    validation_path=directory / "validation",
    pipeline_path=directory / "model",
    experiment=None,
    num_boost_round=100
)

shutil.rmtree(directory)
