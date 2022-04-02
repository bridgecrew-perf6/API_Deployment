import pickle
import pandas as pd
import numpy as np
import json
import joblib

model = joblib.load("model/model.pkl")


def predict(df):


    prediction = model.predict(df.to_numpy())[0]
    price = round(np.exp(prediction), 2)

    return price
