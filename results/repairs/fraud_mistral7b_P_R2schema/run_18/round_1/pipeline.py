import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from mlflow import log_artifact, track, pyfunc
from mlflow.sklearn import save_model
from mlflow.sklearn import load_model

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(data):
    # Preprocessing steps such as filling missing values, encoding categorical variables, etc.
    # For simplicity, I'm assuming there are no missing values and no categorical variables.
    return data[['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']], data['Class']

def train_model(X_train, y_train):
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    track(f"Model Training Time", -tracking_artifact=True)
    mlflow.sklearn.log_model(model, "model")
    return model
