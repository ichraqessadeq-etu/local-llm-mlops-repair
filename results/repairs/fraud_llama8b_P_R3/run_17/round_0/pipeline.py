import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import mlflow.sklearn

def load_data(file_path):
    """Load data from CSV file"""
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def preprocess_data(data):
    """Preprocess data: split into features and target, scale features"""
    try:
        X = data.drop(['Class', 'Time'], axis=1)
        y = data['Class']
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        return X_scaled, y
    except Exception as e:
        print(f"Error preprocessing data: {e}")
        return None, None

def train_model(X_train, y_train):
    """Train RandomForest model"""
    try:
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        mlflow.sklearn.log_model(model, "random_forest")
        return model
    except Exception as e:
        print(f"Error training model: {e}")
        return None
