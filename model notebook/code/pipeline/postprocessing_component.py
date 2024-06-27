

import os
import numpy as np
import json
import joblib

try:
    from sagemaker_containers.beta.framework import encoders, worker
except ImportError:
    # We don't have access to the `worker` package when testing locally.
    # We'll set it to None so we can change the way functions create
    # a response.
    worker = None

def model_fn(model_dir):
    """
    Deserializes the target model and returns the list of fitted categories.
    """

    model = joblib.load(os.path.join(model_dir, "target.joblib"))
    return model.named_transformers_["pre_approved_offeres"].categories_[0]

def input_fn(input_data, content_type):
    if content_type == "application/json":
        return json.loads(input_data)["predictions"]
    
    raise ValueError(f"{content_type} is not supported.")
    

def predict_fn(input_data, model):
    """
    Transforms the prediction into its corresponding binary category.
    """
    # Assuming input_data is the probability of the positive class (class 1)
    predictions = (input_data > 0.5).astype(int)
    return [
        model[prediction] for prediction in predictions
    ]


def output_fn(prediction, accept):
    """
    Formats the prediction output to generate a response.
    """

    if accept == "text/csv":
        return (
            worker.Response(encoders.encode(prediction, accept), mimetype=accept)
            if worker
            else (prediction, accept)
        )

    if accept == "application/json":
        response = [{"prediction": p} for p in prediction]

        # If there's only one prediction, we'll return it as a single object.
        if len(response) == 1:
            response = response[0]

        return (
            worker.Response(json.dumps(response), mimetype=accept)
            if worker
            else (response, accept)
        )

    raise Exception(f"{accept} accept type is not supported.")
