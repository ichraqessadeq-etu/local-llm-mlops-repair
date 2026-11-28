import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from mlflow import log_artifact, pyfunc, track

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(data):
    # Perform any necessary preprocessing such as normalization, encoding, etc.
    # For simplicity, I'll just return the data as is.
    return data['Features'], data['Class']

def train_model(X_train, y_train):
    with mlflow.start_run():
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        pyfunc.log_model(model, "model", registered_model_name="RandomForestFraudDetector")
        track(f"Model training time: {time() - start_time}")
    return model
