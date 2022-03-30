import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import r2_score
import joblib


class Model:
    def __init__(self):
        self.df = pd.read_csv("data/data_model.csv")

    def create_model(self):
        """This function creates a Linear Regression Model for prediction"""

        df = self.df
        # Declare the features and targets

        X = df.drop(["log_price"], axis=1)
        y = df["log_price"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=365
        )

        # Create a scaler object for standardization purposes
        scaler = StandardScaler()
        # Fit the inputs (calculate the mean and standard deviation feature-wise)
        scaler.fit(X_train)
        features_scal = scaler.transform(X_train)
        regressor = LinearRegression()
        regressor.fit(X_train, y_train)
        regressor.score(X_train, y_train)
        y_hat = regressor.predict(X_train)
        scaler.transform(X_test)
        regressor.score(X_test, y_test)

        y_hat = regressor.predict(X_train)
        rmse = np.sqrt(MSE(y_train, y_hat))
        r2 = r2_score(y_train, y_hat)

        print("The model performance for training set")
        print("--------------------------------------")
        print("RMSE is {}".format(rmse))
        print("R2 score is {}".format(r2))
        print("\n")

        # model evaluation for testing set
        y_test_predict = regressor.predict(X_test)
        rmse = np.sqrt(MSE(y_test, y_test_predict))
        r2 = r2_score(y_test, y_test_predict)

        print("The model performance for testing set")
        print("--------------------------------------")
        print("RMSE is {}".format(rmse))
        print("R2 score is {}".format(r2))

        joblib.dump(regressor, "model/model.pkl")

        model_columns = list(X.columns)

        joblib.dump(model_columns, "model/model_columns.pkl")
        print(X.columns)


regression = Model()
#regression.data_cleaning()
regression.create_model()
