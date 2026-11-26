# pipeline.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import mlflow
import logging
from sklearn import ensemble

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
        X = data.drop(['Class', 'Amount'], axis=1)
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
            mlflow.set_tag("model_type", "RandomForest")
            mlflow.set_tag("dataset", "fraud.csv")
            model = ensemble.RandomForestClassifier(n_estimators=100, random_state=42)  # Use ensemble.RandomForestClassifier
            model.fit(X_train, y_train)
            mlflow.sklearn.log_model(model, "model")
            return model
    except Exception as e:
        logging.error(f"Error training model: {e}")
        raise
