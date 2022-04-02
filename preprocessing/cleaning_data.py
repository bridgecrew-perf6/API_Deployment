import pickle
import joblib
import pandas as pd
import numpy as np
import json


def preprocess(data_input):
    expected_outcome = {
        "postcode": {"type": str, "optional": False},
        "kitchen_type": {
            "type": str,
            "optional": True,
            "default": ["Not installed", "Semi equipped",
                "Equipped"],
        },
        "bedroom": {"type": int, "optional": False, "default": []},
        "swimming_pool": {"type": str, "optional": True, "default": ["Yes", "No"]},
        "surface_plot": {"type": int, "optional": True, "default": []},
        "living_area": {"type": int, "optional": False, "default": []},
        "property_type": {
            "type": str,
            "optional": False,
            "default": ["APARTMENT", "HOUSE"],
        },
    }

    for key in expected_outcome.keys():
        if not expected_outcome[key]['optional']:
            if key not in data_input["data"].keys():
                raise ValueError(f'Expected feature {key} is missing')

    for key, value in data_input["data"].items():
        if key not in expected_outcome.keys():
            raise ValueError(f'This feature {key} is not available for this model')
        if type(value) != expected_outcome[key]['type']:
            raise ValueError(f'{key}:{value} should be {expected_outcome[key]["type"]}')
        if expected_outcome[key]['type'] == str and len(expected_outcome[key]['default'])>0:
            if value not in expected_outcome[key]['default']:
                raise ValueError(f'Chosen value is not valid. Please enter a default value from {expected_outcome[key]["default"]}')
        if value == "Not installed":
            data_input['data'][key] = float(0)
        if value == "Semi equipped":
            data_input['data'][key] = float(1)
        if value == "Equipped":
            data_input['data'][key] = float(2)
        if value == "Yes":
            data_input['data'][key] = 1
        if value == "No":
            data_input['data'][key] = 0


    model_columns = pickle.load(open('model/model_columns.pkl', 'rb'))
    df = pd.DataFrame(data_input)
    df = df.T
    df = pd.get_dummies(df, columns=['property_type'])
    df['living_area'] = np.exp(df["living_area"])
    
    return df.reindex(columns=model_columns, fill_value=0)

