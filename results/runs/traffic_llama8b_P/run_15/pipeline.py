import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import mlflow.sklearn
import mlflow

def load_data(file_path):
    """Loads data from CSV file"""
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        mlflow.log_error(f"Error loading data: {e}")
        raise

def preprocess_data(data):
    """Preprocesses data by splitting into features and target, scaling features"""
    try:
        X = data.drop('Congestion', axis=1)
        y = data['Congestion']
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        return X_scaled, y
    except Exception as e:
        mlflow.log_error(f"Error preprocessing data: {e}")
        raise

def train_model(X_train, y_train):
    """Trains a RandomForest classifier"""
    try:
        model = RandomForestClassifier(n_estimators=100)
        model.fit(X_train, y_train)
        mlflow.sklearn.log_model(model, "random_forest")
        return model
    except Exception as e:
        mlflow.log_error(f"Error training model: {e}")
        raise
