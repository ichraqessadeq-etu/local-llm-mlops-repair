import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
from mlflow import log_param, log_metric, log_model

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(data):
    X = data.drop(columns=['Congestion'])
    y = data['Congestion']
    return X, y

def train_model(X_train, y_train):
    with mlflow.start_run():
        log_param("model", "RandomForestClassifier")
        log_param("n_estimators", 100)
        log_param("max_depth", 10)
        log_param("random_state", 42)
        
        model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_train)
        accuracy = accuracy_score(y_train, y_pred)
        log_metric("train_accuracy", accuracy)
        
        log_model(model, "random_forest_model")
        
        return model
