from flask import Flask, request, jsonify
from predict.prediction import prediction
import json
import os


from preprocessing.cleaning_data import preprocess

app = Flask(__name__)


@app.route("/", methods=["GET"])
def check_api():
    """Checks if the server is alive"""
    return "Alive!!!"


@app.route("/predict", methods=["GET", "POST"])
def predict():

    if request.method == "POST":
        json_ = request.get_json()
        prediction_df = preprocess(json_)
        price = prediction(prediction_df)
        return {"prediction": price}

    else:
        expected_outcome = {
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
                "default": ["Apartment", "House"],
            },
        }
        return expected_outcome


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", threaded=True, port=port)
