import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import mlflow
import mlflow.sklearn

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(data):
    return (data[features], data['Anomaly'])

def train_model(X_train, y_train):
    with mlflow.start_run():
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        mlflow.sklearn.log_model(model, 'anomaly_detector')
    return model