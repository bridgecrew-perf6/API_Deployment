import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import r2_score
import joblib


class Model():
    
    def __init__(self):
        self.df = pd.read_csv('data/20220208_Final_result.csv')


    def data_cleaning(self) -> pd.DataFrame:
        """This function cleans the data for machine learning model"""

        df = self.df

        # changing the datatype of Price column to numeric
        df = df[pd.to_numeric(df["Price"], errors="coerce").notnull()]
        df = df.astype({"Price": float}, errors="raise")

        # dropping dublicates base on the Immoweb ID as those ID's should be unique
        df = df.drop_duplicates(["Immoweb ID"], keep="last")

        # dropping columns which will not be used in ml model
        df = df.drop(
            columns=["Tenement building", "How many fireplaces?", "Garden orientation"]
        )

        building_condition_map = {
            "As new": 6,
            "Just renovated": 5,
            "Good": 4,
            "To be done up": 3,
            "To renovate": 2,
            "To restore": 1,
        }
        df = df.applymap(
            lambda s: building_condition_map.get(s) if s in building_condition_map else s
        )
        df["Building condition"] = df["Building condition"].fillna(2)
        print(df["Building condition"].isnull().sum())

        Kit_type_dict = {
            "USA uninstalled": 0,
            "Not installed": 0,
            "Installed": 1,
            "USA installed": 1,
            "Semi equipped": 1,
            "USA semi equipped": 1,
            "Hyper equipped": 2,
            "USA hyper equipped": 2,
        }

        df = df.replace(Kit_type_dict)
        df["Kitchen type"] = df["Kitchen type"].fillna(0)
        
        df["Furnished"] = df["Furnished"].fillna("No")
        df["Furnished"] = df["Furnished"].apply(lambda v: 0 if v == "No" else 1)

        df["Bedrooms"] = df["Bedrooms"].fillna(2).astype(int)

        # Fill missing values with value 0
        df["Swimming pool"].fillna(0, inplace=True)
        df["Swimming pool"] = df["Swimming pool"].apply(lambda v: 0 if v == "No" else 1)
        df["Swimming pool"].isnull().sum()

        # Fill empty values with 0 all missing values correspond to the Apartement
        df["Surface of the plot"].fillna(0, inplace=True)


        # get ['number of frontages'] with values and calc mean
        selected_rows = df[~df["Number of frontages"].isnull()]
        mean_num_of_frontages = selected_rows["Number of frontages"].mean(axis=0).round(0)
        mean_num_of_frontages

        # fill mean value to missing value
        df["Number of frontages"] = df["Number of frontages"].fillna(mean_num_of_frontages)
        # changing data type as int
        df["Number of frontages"] = df["Number of frontages"].astype(int)


        ## Garden/Garden Surface & Terrace/Surface

        df.loc[df.Garden == 1, "garden_area" ] = df.loc[df.Garden == 1, "garden_area"].fillna(df['garden_area'].median())
        df.loc[df.Garden == 0, "garden_area" ] = df.loc[df.Garden == 0, "garden_area"].fillna(0)

        df.loc[df.Terrace == 1, "terrace_area" ] = df.loc[df.Terrace == 1, "terrace_area"].fillna(df['terrace_area'].median())
        df.loc[df.Garden == 0, "terrace_area" ] = df.loc[df.Terrace == 0, "terrace_area"].fillna(0)
        # As it was mentioned before these columns will not be used for further analysis
        df = df.drop(columns=["garden_area", "terrace_area"])

        df.drop(["Furnished"], axis=1)

        ### Dealing with Outliers

        # This variable is equal to the 99th percentile of the 'Price' variable
        q = df["Price"].quantile(0.99)
        # Then we can create a new df, with the condition that all prices must be below the 99 percentile of 'Price'
        data_1 = df[df["Price"] < q]
        # In this way we have removed the top 1% of the data about 'Price'

        # #### Living area

        q = data_1["Living area"].quantile(0.99)
        data_2 = data_1[data_1["Living area"] < q]

        #### Surface of the plot


        q = data_2["Surface of the plot"].quantile(0.99)
        data_3 = data_2[data_2["Surface of the plot"] < q]


        #### Number of frontages

        q = data_3["Number of frontages"].quantile(0.99)
        data_4 = data_3[data_3["Number of frontages"] < q]

        #### Bedrooms

        q = data_4["Bedrooms"].quantile(0.99)
        data_5 = data_4[data_4["Bedrooms"] < q]

        data_cleaned = data_5.reset_index(drop=True)
        data_cleaned["log_area"] = data_cleaned["Living area"].map(np.log)

        # Then we add it to our data frame
        data_cleaned["log_price"] = data_cleaned["Price"].map(np.log)
        data_cleaned

        data_cleaned['log_price'].max()
        data_cleaned = data_cleaned.drop(["Price", 'Living area'], axis=1)
        data_no_multicollinearity = data_cleaned.drop(
            ["Number of frontages", "Building condition"], axis=1
        )
        ### Categorical data encoding
        data_no_multicollinearity["Post code"].astype(str)
        post_code_stat = data_no_multicollinearity["Post code"].value_counts(ascending=False)
        pc_stat_less_10 = post_code_stat[post_code_stat <= 10]
        data_post_code = data_no_multicollinearity.copy()
        data_post_code["Post code"] = data_no_multicollinearity["Post code"].apply(
            lambda x: "other" if x in pc_stat_less_10 else x
        )
        len(data_post_code["Post code"].unique())
        data_no_pro_type = data_post_code.drop(
            columns=["Immoweb ID", "Property type"]
        )
        data_with_dummies = pd.get_dummies(data_no_pro_type, drop_first=True)
        df = data_with_dummies

        return df

    
    def create_model(self):
        """ This function creates a Linear Regression Model for prediction"""
        
        self.df = self.data_cleaning()
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

        joblib.dump(regressor, 'model/model.pkl')

        model_columns =list(df.columns)

        joblib.dump(model_columns, 'model/model_columns.pkl')

regression = Model()
regression.data_cleaning()
regression.create_model()
