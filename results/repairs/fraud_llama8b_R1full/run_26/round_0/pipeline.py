import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import mlflow
import mlflow.sklearn

def load_data(file_path):
    """Loads the data from the CSV file"""
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        raise

def preprocess_data(data):
    """Preprocesses the data by scaling and splitting into features and target"""
    try:
        X = data.drop(['Class'], axis=1)
        y = data['Class']
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test
    except Exception as e:
        print(f"Error preprocessing data: {e}")
        raise

def train_model(X_train, y_train):
    """Trains a RandomForest classifier on the preprocessed data"""
    try:
        with mlflow.start_run():
            mlflow.set_tag("model_type", "RandomForest")
            mlflow.set_tag("target_column", "Class")
            mlflow.log_param("n_estimators", 100)
            mlflow.log_param("max_depth", 10)
            model = RandomForestClassifier(n_estimators=100, max_depth=10)
            model.fit(X_train, y_train)
            mlflow.sklearn.log_model(model, "random_forest_model")
            return model
    except Exception as e:
        print(f"Error training model: {e}")
        raise
