from flask import Flask, request, jsonify
import json
import os
import pandas as pd
from predict.prediction import predict

from preprocessing.cleaning_data import preprocess

app = Flask(__name__)


@app.route("/", methods=["GET"])
def check_api():
    """Checks if the server is alive"""
    return "Alive!!!"


@app.route("/predict", methods=["GET"])
def input_format():

    expected_outcome = """
    {
        "data": {

        "postcode": {"type": int, "optional": False},
        "kitchen_type": {
            "type": str,
            "optional": True,
            "default": ["Not installed", "Semi equipped", "Equipped"],
        },
        "bedroom": {"type": int, "optional": False, "default": []},
        "swimming_pool": {"type": str, "optional": True, "default": ["Yes", "No"]},
        "surface_plot": {"type": int, "optional": True, "default": []},
        "living_area": {"type": int, "optional": False, "default": []},
        "property_type": {
            "type": str,
            "optional": False,
            "default": ["APARTMENT", "HOUSE"],
        }
    }"""
    return f"Your input should be in this format: {expected_outcome}"


@app.route("/predict", methods=["POST"])
def respond():
    json_ = request.get_json()
    df = pd.DataFrame(json_)
    if "data" not in json_.keys():
        return {"error" : "data not is there"}
    prediction_df = preprocess(json_)
    price = predict(prediction_df)
    return {"prediction": price}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", threaded=True, port=port)
