# pipeline.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import mlflow
import mlflow.sklearn
import logging

def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise

def preprocess_data(data):
    try:
        # Convert clouds_all to numerical type
        data['clouds_all'] = pd.Categorical(data['clouds_all']).codes
        
        # Handle missing values in clouds_all
        data['clouds_all'] = data['clouds_all'].fillna(0)
        
        X = data.drop(['Congestion'], axis=1)
        y = data['Congestion']
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logging.error(f"Error preprocessing data: {e}")
        raise

def train_model(X_train, y_train):
    try:
        with mlflow.start_run():
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            mlflow.sklearn.log_model(model, "random_forest")
            return model
    except Exception as e:
        logging.error(f"Error training model: {e}")
        raise
