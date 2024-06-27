%%ipytest
#| code-fold: true

from pipeline.preprocessing_component import input_fn, predict_fn, output_fn, model_fn,feature_columns,TARGET_COLUMN
from processing.script import preprocess
from processing.script import (
    _read_data_from_input_csv_files, 
    convert_to_lowercase, 
    handle_missing_values, 
    extract_and_transform_features, 
    adjust_numeric_features, 
    add_new_feature, 
    drop_column
)
import os
import shutil
import tempfile
from pathlib import Path
import pandas as pd
import pytest
import ipytest
import tarfile
import numpy as np

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

def preprocess_for_test(base_directory):
    df = _read_data_from_input_csv_files(base_directory)
    
    df = convert_to_lowercase(df)  
    df = handle_missing_values(df)                         
    df = extract_and_transform_features(df)                
    df = adjust_numeric_features(df)                       
    df = add_new_feature(df)                               
    df = drop_column(df)
    
    # Ensure columns expected to be numeric are converted to numeric types
    numeric_columns = ['loan_amnt', 'installment', 'annual_inc', 'dti', 'revol_bal']
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')

    # Ensure pincode column is treated as a string
    df['pincode'] = df['pincode'].astype(str)

    # Save the processed data
    processed_directory = base_directory / "processed"
    processed_directory.mkdir(parents=True, exist_ok=True)
    df.to_csv(processed_directory / "data.csv", index=False)

    return df



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

def test_output_returns_xgboost_ready_input():
    # Simulated processed data that would be passed to the model
    processed_sample_data = np.array([
        [8.96200721, 5.541420396, 8.016647877, 2.313525033, 5.424950017, 1, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 3, 0, 0, 0, 0, 0]
    ])
    
    # Simulated model output (predictions)
    prediction = np.array([
        [0.8, 0.2],  # Example prediction probabilities for binary classification
        [0.1, 0.9]
    ])
    
    # Invoke the output function
    response, mimetype = output_fn(prediction, "application/json")
    
    # Expected response
    expected_response = {
        "instances": [
            [0.8, 0.2],
            [0.1, 0.9]
        ]
    }
    
    # Check the response content and MIME type
    assert response == expected_response, "Response content does not match expected output"
    assert mimetype == "application/json", "Response MIME type is not 'application/json'"



def test_predict_transforms_data(directory):
    # Path to the sample data file containing 100 rows
    sample_data_filepath = directory.parent / "input" / "data.csv"
    
    # Read one row from the sample data file
    raw_df = pd.read_csv(sample_data_filepath, nrows=1)
    
    # Create a temporary directory to store the single row raw input CSV
    with tempfile.TemporaryDirectory() as tempdir:
        base_directory = Path(tempdir)
        input_csv_path = base_directory / "input" / "data.csv"
        input_csv_path.parent.mkdir(parents=True, exist_ok=True)
        raw_df.to_csv(input_csv_path, index=False)
        
        # Run the new preprocessing function for the test
        preprocess_for_test(base_directory)
        
        # Load the processed data
        processed_csv_path = base_directory / "processed" / "data.csv"
        
        # Add debug statement to verify the path
        assert processed_csv_path.exists(), f"Processed data file not found at {processed_csv_path}"
        
        df = pd.read_csv(processed_csv_path)

        # Ensure pincode column is treated as string
        df['pincode'] = df['pincode'].astype(str)

        # Load the model
        model = model_fn(directory.as_posix())
        
        # Make predictions
        response = predict_fn(df, model)
        
        # Validate the response
        assert type(response) is np.ndarray

def test_predict_returns_none_if_invalid_input(directory):
    # Invalid input data
    input_data = """
    Invalid, 36 months, 11.44, 329.48, B, B4, Marketing, 10+ years, RENT, 117000.0, Not Verified, Jan-2015, Fully Paid, vacation, Vacation, 26.24, Jun-1990, 16.0, 0.0, 36369.0, 41.8, 25.0, w, INDIVIDUAL, 0.0, 0.0, 0174 Michelle Gateway\r\nMendozaberg, OK 22690
    """
    
    model = model_fn(directory.as_posix())
    try:
        df = input_fn(input_data, "text/csv")
    except ValueError as e:
        df = None
    
    # df should be None if input is invalid
    assert df is None or predict_fn(df, model) is None, "predict_fn should return None for invalid input"
