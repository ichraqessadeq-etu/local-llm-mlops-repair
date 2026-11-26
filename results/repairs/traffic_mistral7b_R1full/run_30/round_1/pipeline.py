import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.sklearn
import numpy as np

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(data):
    # Perform any necessary data preprocessing like feature scaling, encoding, etc.
    # For simplicity, I'll just replace the 'Clouds' column with a number.
    data['Clouds_all'].replace('Clouds', np.nan, inplace=True)
    return data, data['Congestion']

def train_model(X_train, y_train):
    with mlflow.start_run():
        model = RandomForestClassifier(random_state=42)  # Seed the random number generator
        model.fit(X_train, y_train)
        mlflow.sklearn.log_model(model, "random-forest")

    return model
