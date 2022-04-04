import pickle
import pandas as pd
import numpy as np
import joblib

#Linear regression model
model = joblib.load("model/model.pkl")

def predict(df):

    """This function takes user input as parameter and returns the prediction
    for given features
    param : df 
    """

    #Getting the O element(price) of nested list
    prediction = model.predict(df.to_numpy())[0]
    #Converting the log price to exponential value
    price = round(np.exp(prediction), 2)

    return price
