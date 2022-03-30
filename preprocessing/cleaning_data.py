import pickle
import pandas as pd


def preprocess(json_data):

    expected_outcome = {
        "postcode": {"type": int, "optional": False},
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
            "default": ["Apartment", "House"],
        },
    }

    for key in expected_outcome.keys():
        if expected_outcome[key]['optional']:
            if key not in json_data.keys():
                continue
            raise ValueError(f'Expected feature {key} is missing. Please ')

    for key, value in json_data.items():
        if key not in expected_outcome.keys():
            raise ValueError(f'This feature {key} is not available for this model')
        if type(value) != expected_outcome[key]['type']:
            raise ValueError(f'{key}:{value} should be {expected_outcome[key]["type"]}')
        if expected_outcome['type'] == str and len(expected_outcome['type'])>0:
            if value not in expected_outcome[key]['default']:
                raise ValueError(f'Chosen value is not valid. Please enter a default value from {expected_outcome[key]["default"]}')
        if value == "Not installed":
            value = float(0)
        if value == "Semi equipped":
            value = float(1)
        if value == "Equipped":
            value = float(2)
        if value == "Yes":
            value = 1
        if value == "No":
            value = 0

    df = pd.DataFrame(json_data[expected_outcome], index=0)

    df = pd.get_dummies(df)

    model_columns = pickle.load(open('model/model_columns.pkl'), 'rb')

    df = df.reindex(columns=model_columns, fill_value=0)

    return df

