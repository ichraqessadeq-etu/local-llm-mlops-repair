import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from mlflow import PyFuncModel, log_model, log_artifact

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(data):
    # Perform any necessary preprocessing such as normalization, encoding, etc.
    # For simplicity, I'll just drop a column here.
    data.drop('V2', axis=1, inplace=True)
    return data.drop('Amount', axis=1), data['Amount']

def train_model(X_train, y_train):
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model
