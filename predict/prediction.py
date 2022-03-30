import pickle
import pandas as pd
import numpy as np

def prediction(preprocessed_data):
    
    model = pickle.load('model/model.pkl')

    price_log = model.predict(preprocessed_data)[0]
    price = round(np.exp(price_log), 2)

    return price

    