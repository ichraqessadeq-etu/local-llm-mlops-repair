import mlflow
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(data):
    data.dropna(inplace=True)
    X = data.drop('Anomaly', axis=1)
    y = data['Anomaly']
    return (X, y)

def train_model(X_train, y_train):
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    try:
        mlflow.pyfunc.log_model(pyfunc.DistributedModel.from_flask(model, 'RandomForestAnomalyDetector'), artifact_path='model')
    except Exception as e:
        print(f"Error while logging model: {e}")
