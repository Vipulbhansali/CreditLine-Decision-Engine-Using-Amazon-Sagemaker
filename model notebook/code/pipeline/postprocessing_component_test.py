%%ipytest
#| code-fold: true


import numpy as np

from pipeline.postprocessing_component import predict_fn, output_fn

def test_predict_returns_prediction_as_binary_category():
    input_data = np.array([0.8, 0.3, 0.6])

    categories = ["No", "Yes"]
    
    response = predict_fn(input_data, categories)
    
    assert response == [
        "Yes",  # 0.8 > 0.5, hence "Yes"
        "No",   # 0.3 <= 0.5, hence "No"
        "Yes"   # 0.6 > 0.5, hence "Yes"
    ]

def test_output_does_not_return_array_if_single_prediction():
    prediction = ["Yes"]
    response, _ = output_fn(prediction, "application/json")

    assert response["prediction"] == "Yes"

def test_output_returns_array_if_multiple_predictions():
    prediction = ["Yes", "No"]
    response, _ = output_fn(prediction, "application/json")

    assert len(response) == 2
    assert response[0]["prediction"] == "Yes"
    assert response[1]["prediction"] == "No"

%%writefile {CODE_FOLDER}/pipeline/postprocessing_component_test.py
