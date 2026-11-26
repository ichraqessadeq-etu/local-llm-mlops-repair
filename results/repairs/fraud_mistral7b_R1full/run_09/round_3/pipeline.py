import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import mlflow
import mlflow.sklearn
from sklearn.utils.multiclass import type_of_target
from sklearn.preprocessing import LabelBinarizer

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(data):
    # Add any necessary preprocessing steps like feature scaling, encoding, etc.
    # For simplicity, I'll just return the data as it is.
    return data, data['Amount']

def preprocess_target(y):
    lb = LabelBinarizer()
    y_bin = lb.fit_transform(y)
    return y_bin

def train_model(X_train, y_train, random_state=42):
    with mlflow.start_run():
        y_type = type_of_target(y_train)
        if y_type != "continuous":
            y_bin = preprocess_target(y_train)
            model = RandomForestClassifier(random_state=random_state)
            model.fit(X_train, y_bin)
            mlflow.sklearn.log_model(model, "random-forest")
        else:
            raise ValueError("Unknown label type: continuous")

    return model
