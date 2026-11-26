import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import mlflow
import mlflow.sklearn
import logging

logging.basicConfig(level=logging.INFO)

def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise

def preprocess_data(data):
    try:
        X = data.drop(['Class', 'Time'], axis=1)
        y = data['Class']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        return X_train_scaled, X_test_scaled, y_train, y_test
    except Exception as e:
        logging.error(f"Error preprocessing data: {e}")
        raise

def train_model(X_train, y_train):
    try:
        with mlflow.start_run():
            mlflow.log_param("model_type", "RandomForestClassifier")
            mlflow.log_param("n_estimators", 100)
            mlflow.log_param("max_depth", 5)
            mlflow.log_param("min_samples_split", 2)
            mlflow.log_param("min_samples_leaf", 1)
            mlflow.log_param("max_features", "auto")
            mlflow.log_param("random_state", 42)
            model = RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_split=2, min_samples_leaf=1, max_features="auto", random_state=42)
            model.fit(X_train, y_train)
            mlflow.sklearn.log_model(model, "random_forest_model")
            return model
    except Exception as e:
        logging.error(f"Error training model: {e}")
        raise
