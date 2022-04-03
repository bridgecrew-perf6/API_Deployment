from flask import Flask, request, jsonify
import json
import os
import pandas as pd
from predict.prediction import predict
from preprocessing.cleaning_data import preprocess
from preprocessing.validator import input_validator

app = Flask(__name__)


@app.route("/", methods=["GET"])
def check_api():
    """Checks if the server is alive"""
    return "Alive!!!"


@app.route("/predict", methods=["GET"])
def input_format():

    with open("preprocessing/template.json") as file:
        output = json.load(file)

        # Beautify output using Flask templates
        return jsonify(output), 200


@app.route("/predict", methods=["POST"])
def respond():
    json_ = request.get_json().get('data')
    validation = input_validator(json_)
    if validation == "Excellent!":    
        prediction_df = preprocess(json_)
        price = predict(prediction_df)
        return jsonify({"prediction": price, "error": None}), 200
    else:
        return jsonify({"errors": validation, "prediction": None}), 400


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(debug=False, host="0.0.0.0", threaded=True, port=port)
