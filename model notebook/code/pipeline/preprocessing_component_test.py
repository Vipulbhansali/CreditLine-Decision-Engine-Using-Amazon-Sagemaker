
%%ipytest
#| code-fold: true

from pipeline.preprocessing_component import input_fn, predict_fn, output_fn, model_fn,feature_columns,TARGET_COLUMN
from processing.script import preprocess

import os
import shutil
import tempfile
from pathlib import Path
import pandas as pd
import pytest
import ipytest
import tarfile

ipytest.autoconfig()

DATA_FILEPATH = r"C:\Users\Vipul\CreditLine-Decision-Engine-Using-Amazon-Sagemaker\credit.csv"

sample_csv_row = """10000.0,36 months,11.44,329.48,B,B4,Marketing,10+ years,RENT,117000.0,Not Verified,Jan-2015,Fully Paid,vacation,Vacation,26.24,Jun-1990,16.0,0.0,36369.0,41.8,25.0,w,INDIVIDUAL,0.0,0.0,0174 Michelle Gateway\r\nMendozaberg, OK 22690"""

sample_json_row = {
    "loan_amnt": 10000.0,
    "term": "36 months",
    "int_rate": 11.44,
    "installment": 329.48,
    "grade": "B",
    "sub_grade": "B4",
    "emp_title": "Marketing",
    "emp_length": "10+ years",
    "home_ownership": "RENT",
    "annual_inc": 117000.0,
    "verification_status": "Not Verified",
    "issue_d": "Jan-2015",
    "loan_status": "Fully Paid",
    "purpose": "vacation",
    "title": "Vacation",
    "dti": 26.24,
    "earliest_cr_line": "Jun-1990",
    "open_acc": 16.0,
    "pub_rec": 0.0,
    "revol_bal": 36369.0,
    "revol_util": 41.8,
    "total_acc": 25.0,
    "initial_list_status": "w",
    "application_type": "INDIVIDUAL",
    "mort_acc": 0.0,
    "pub_rec_bankruptcies": 0.0,
    "address": "0174 Michelle Gateway\r\nMendozaberg, OK 22690",
     "pincode": '22690',  # Add this column
    "revol_util_bins": "40-50"  # Add this column
}


@pytest.fixture(scope="function", autouse=False)
def directory():
    base_directory = Path(tempfile.mkdtemp())
    input_directory = base_directory / "input"
    input_directory.mkdir(parents=True, exist_ok=True)

    # Load the first 100 rows from the original data file and save it to a temporary file
    data = pd.read_csv(DATA_FILEPATH, nrows=100)
    sample_data_filepath = input_directory / "data.csv"
    data.to_csv(sample_data_filepath, index=False)
    
    preprocess(base_directory=base_directory)
    
    with tarfile.open(base_directory / "model" / "model.tar.gz") as tar:
        tar.extractall(path=base_directory / "model")
    
    yield base_directory / "model"
    
    shutil.rmtree(base_directory)

def test_model_fn(directory):
    model = model_fn(directory)
    assert model is not None, "Model should not be None"

def test_input_fn_csv():
    df = input_fn(sample_csv_row, "text/csv")
    assert isinstance(df, pd.DataFrame), "Output should be a DataFrame"
    assert list(df.columns) == feature_columns, "Columns do not match feature columns"

def test_input_fn_json():
    input_data = json.dumps(sample_json_row)
    df = input_fn(input_data, "application/json")
    assert isinstance(df, pd.DataFrame), "Output should be a DataFrame"
    assert list(df.columns) == feature_columns, "Columns do not match feature columns"
    assert TARGET_COLUMN not in df.columns, "Target column should be removed"


def test_predict_fn(directory):
    model = model_fn(directory)
    df = pd.DataFrame([sample_json_row])
    transformed_data = predict_fn(df, model)
    assert transformed_data is not None, "Transformed data should not be None"


def test_output_fn():
    prediction = pd.DataFrame([sample_json_row])
    response, mimetype = output_fn(prediction, "application/json")
    
    # If response is already a dictionary, no need to parse it again
    if isinstance(response, str):
        response_json = json.loads(response)
    else:
        response_json = response
    
    assert "instances" in response_json, "Response should contain 'instances' key"
