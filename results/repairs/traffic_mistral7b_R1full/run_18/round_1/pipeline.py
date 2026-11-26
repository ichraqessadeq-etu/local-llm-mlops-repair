import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import mlflow
import numpy as np

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(data):
    # Add any necessary preprocessing steps here, such as normalization, encoding, etc.
    # For simplicity, I'll just return the original data.
    return data, data['Congestion']

def train_model(X_train, y_train):
    random_state = np.random.randint(0, 100)
    with mlflow.start_run():
        model = RandomForestClassifier(random_state=random_state)
        model.fit(X_train, y_train)
        mlflow.log_artifact("model", model)
    return model
