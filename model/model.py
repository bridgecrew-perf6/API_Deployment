import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib


def create_model():
    """This function creates a Linear Regression Model for prediction"""

    df = pd.read_csv("data/data_model.csv")
    # Declare the features and targets

    X = df.drop(["log_price"], axis=1)
    y = df["log_price"]

    # Splitting the data set as "train" and "test"
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=365
    )

    # Create a scaler object for standardization purposes
    scaler = StandardScaler()
    # Fit and transform the inputs in the scaler object
    scaler.fit(X_train)
    features_scal = scaler.transform(X_train)

    #Creating the model and train it
    regressor = LinearRegression().fit(X_train, y_train)

    #Generating pickle file to use model for deployment
    joblib.dump(regressor, "model/model.pkl")

    #Saving model columns to compare & adjust with the user input
    model_columns = list(X.columns)
    joblib.dump(model_columns, "model/model_columns.pkl")


create_model()
