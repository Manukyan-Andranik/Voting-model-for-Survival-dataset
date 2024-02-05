import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
class Preprocessor:
    def __init__(self, scaler = StandardScaler()):
        self.scaler = scaler
        self.data = None
        self.columns = []
        self.null_values = {}
    def fit(self, data,  target = "In-hospital_death", test_mode = True):
        self.columns = data.columns.tolist()
        # Remove unnecessary columns
        columns_for_remove = ["recordid", "SAPS-I", "SOFA", "Length_of_stay", "Survival", "Cholesterol_first", "TroponinI_first", "TroponinT_first", "Cholesterol_last", "TroponinI_last", "TroponinT_last", "RespRate_first", "RespRate_last", "RespRate_lowest", "RespRate_highest", "RespRate_median"]
        for col in columns_for_remove:
            try:
                self.columns.remove(col)
            except:
                continue
        # Fit scaler on features
        if not test_mode:
            X = data.drop(target, axis = 1)
            self.scaler.fit(X)
        else:
            self.scaler.fit(data)


    def transform(self, data,  target = "In-hospital_death", test_mode = False):
        if test_mode:
            # Handle missing values in test data
            columns = self.columns.copy()
            columns.remove(target)
            X = data[columns].copy()
            for col in columns:
                if X[col].isnull().any():
                    X[col].fillna(self.null_values[col], inplace=True)

            return self.scaler.transform(X)

        else:
            # Handle missing values in train data
            X = data[self.columns].copy()

            for col in self.columns:
                if X[col].isnull().any():
                    X_copy = X.copy()
                    if col in ["MechVentLast8Hour", "Gender"]:
                        X[col].fillna(X[col].mode().iloc[0], inplace=True)
                        self.null_values[col] = X_copy[col].mode().iloc[0]
                    else:
                        X[col].fillna(X_copy[col].mean(), inplace=True)
                        self.null_values[col] = X_copy[col].mean()

            # Oversample the minority class
            y = X[target]
            X_train = X.drop(target, axis = 1)

            X1 = X_train[y == 1].copy()
            n_class1 = len(X1)
            n_sample = np.sum(y == 0) // n_class1

            for _ in range(n_sample):
                X_train = pd.concat([X_train, X1], axis=0)

            y_train = pd.concat([y, pd.Series(np.ones(n_sample * n_class1))])
            X_train = self.scaler.fit_transform(X_train)
            return (X_train, y_train)

