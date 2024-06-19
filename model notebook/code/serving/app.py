# | filename: app.py
# | code-line-numbers: true

import tarfile
import tempfile
import numpy as np

from flask import Flask, request, jsonify
from pathlib import Path
import xgboost as xgb

MODEL_PATH = Path(__file__).parent

class Model:
    model = None
    feature_names = [
        "feature_0", "feature_1", "feature_2", "feature_3", "feature_4", "feature_5",
        "feature_6", "feature_7", "feature_8", "feature_9", "feature_10", "feature_11",
        "feature_12", "feature_13", "feature_14", "feature_15", "feature_16", "feature_17",
        "feature_18", "feature_19", "feature_20", "feature_21", "feature_22", "feature_23"
    ]

    def load(self):
        """
        Extracts the model package and loads the model in memory
        if it hasn't been loaded yet.
        """
        if not Model.model:
            with tempfile.TemporaryDirectory() as directory:
                with tarfile.open(MODEL_PATH / "model.tar.gz") as tar:
                    tar.extractall(path=directory)
                bst = xgb.Booster()
                bst.load_model(Path(directory) / "xgboost_model.json")
                Model.model = bst

    def predict(self, data):
        """
        Generates predictions for the supplied data.
        """
        self.load()
        dmatrix = xgb.DMatrix(data, feature_names=self.feature_names)
        return Model.model.predict(dmatrix)

app = Flask(__name__)
model = Model()

@app.route("/predict/", methods=["POST"])
def predict():
    data = np.array(request.json).astype(np.float32)
    data = np.expand_dims(data, axis=0)
    prediction = ((model.predict(data)) > 0.5).astype(int)
    return jsonify({"prediction": prediction.tolist()})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
