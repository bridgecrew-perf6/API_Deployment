import pickle
import pandas as pd
import numpy as np
import pgeocode


def preprocess(data_input):

    build_cond_map = {
        "As new": 6,
        "Just renovated": 5,
        "Good": 4,
        "To be done up": 3,
        "To renovate": 2,
        "To restore": 1,
    }
    kitchen_type_map = {"Not installed": 0, "Semi equipped": 1, "Equipped": 2}
    for key, value in data_input.items():
        for k, v in build_cond_map.items():
            if value == k:
                data_input[key] = v
        for type, num in kitchen_type_map.items():
            if value == type:
                data_input[key] = num

    df = pd.DataFrame(data_input, index=[0])
    df.replace(to_replace="Yes", value=1, inplace=True)
    df.replace(to_replace="No", value=0, inplace=True)
    df["living_area"] = df["living_area"].astype(int)
    df["bedroom"] = df["bedroom"].astype(int)
    df["surface_plot"] = df["surface_plot"].astype(int)

    df["living_area"] = df["living_area"].map(np.log)
    nomi = pgeocode.Nominatim("be")
    df["city"] = nomi.query_postal_code(df["postcode"])["community_name"]
    df.drop(columns=["postcode"])
    df = pd.get_dummies(df, columns=["property_type", "city"])
    model_columns = pickle.load(open("model/model_columns.pkl", "rb"))
    df = df.reindex(columns=model_columns, fill_value=0)

    return df
