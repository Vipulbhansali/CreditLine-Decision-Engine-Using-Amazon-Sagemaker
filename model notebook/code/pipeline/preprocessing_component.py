#| filename: preprocessing_component.py
#| code-line-numbers: true

import os
import pandas as pd
import json
import joblib

from io import StringIO

try:
    from sagemaker_containers.beta.framework import worker
except ImportError:
    # We don't have access to the `worker` package when testing locally.
    # We'll set it to None so we can change the way functions create
    # a response.
    worker = None

TARGET_COLUMN = 'pre_approved_offer'

feature_columns = [
    "loan_amnt",
    "term",
    "int_rate",
    "installment",
    "grade",
    "sub_grade",
    "emp_title",
    "emp_length",
    "home_ownership",
    "annual_inc",
    "verification_status",
    "issue_d",
    "loan_status",
    "purpose",
    "title",
    "dti",
    "earliest_cr_line",
    "open_acc",
    "pub_rec",
    "revol_bal",
    "revol_util",
    "total_acc",
    "initial_list_status",
    "application_type",
    "mort_acc",
    "pub_rec_bankruptcies",
    "address"
]

def model_fn(model_dir):
    """
    Deserializes the model that will be used in this container.
    """

    return joblib.load(os.path.join(model_dir, "features.joblib"))


def input_fn(input_data, content_type):
    """
    Parses the input payload and creates a Pandas DataFrame.

    This function will check whether the target column is present in the
    input data and will remove it.
    """

    if content_type == "text/csv":
        df = pd.read_csv(StringIO(input_data), header=None, skipinitialspace=True)

        # If we find an extra column, it's probably the target
        # feature, so let's drop it. We'll assume the target
        # is always the first column,
        if len(df.columns) == len(feature_columns) + 1:
            df = df.drop(df.columns[-1], axis=1)

        df.columns = feature_columns
        return df

    if content_type == "application/json":
        df = pd.DataFrame([json.loads(input_data)])

        if TARGET_COLUMN in df.columns:
            df = df.drop(TARGET_COLUMN, axis=1)
            
         # Filter out any columns not in feature_columns
        df = df[feature_columns]
        return df
        

    raise ValueError(f"{content_type} is not supported!")

def predict_fn(input_data, model):
    """
    Preprocess the input using the transformer.
    """

    try:
        return model.transform(input_data)
    except ValueError as e:
        print("Error transforming the input data", e)
        return None

import json

def output_fn(prediction, accept):
    """
    Formats the prediction output to generate a response.

    The default accept/content-type between containers for serial inference
    is JSON. This function ensures that the prediction output from XGBoost
    is formatted as a JSON object.
    """

    if prediction is None:
        raise Exception("There was an error transforming the input data")

    # Convert the predictions to a list if they are not already
    instances = prediction.tolist() if hasattr(prediction, 'tolist') else prediction
    response = {"instances": instances}
    
    # Returning the response in the specified accept format
    return (
        worker.Response(json.dumps(response), mimetype=accept)
        if worker
        else (response, accept)
    )
