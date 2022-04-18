import pickle
import pandas as pd
import numpy as np
import pgeocode

#Open the columns name which were used to train the model
model_columns = pickle.load(open("model/model_columns.pkl", "rb"))

def preprocess(data_input):
    """This function gets user input after validator proces in Json format, clean it and 
    give the same format which was used to train the model."""

    #Converting the categorical features to ordinal values
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

    #Create a data frame from json data
    df = pd.DataFrame(data_input, index=[0])

    #Converting the categorical features to boolean values
    df.replace(to_replace="Yes", value=1, inplace=True)
    df.replace(to_replace="No", value=0, inplace=True)

    #Setting data type to int for numerical values
    df["living_area"] = df["living_area"].astype(int)
    df["bedroom"] = df["bedroom"].astype(int)
    df["surface_plot"] = df["surface_plot"].astype(int)

    #Getting the log value of living area as in the model it was used as log value
    df["living_area"] = df["living_area"].map(np.log)

    #Converting postal code to Ciyt name with the help of pgeocode
    nomi = pgeocode.Nominatim("be")
    df["city"] = nomi.query_postal_code(df["postcode"])["community_name"]

    #Drop the postcode column as the model use it as city name
    df = df.drop(columns=["postcode"])

    #Converting the categorical values to numerical values with dummies method
    df = pd.get_dummies(df, columns=["property_type", "city"])

    #Compare the the features in the user input and the model columns name
    df = df.reindex(columns=model_columns, fill_value=0)
    #Reindex dataframe on the base of model columns if column is missing 
    #replace its value as 0

    return df
