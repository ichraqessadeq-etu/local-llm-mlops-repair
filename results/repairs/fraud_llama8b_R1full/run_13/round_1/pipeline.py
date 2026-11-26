# pipeline.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import mlflow
import mlflow.sklearn

def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        raise

def preprocess_data(data):
    try:
        # Drop unnecessary columns
        data.drop(['Time'], axis=1, inplace=True)
        
        # Scale features using StandardScaler
        scaler = StandardScaler()
        data[['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28']] = scaler.fit_transform(data[['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28']])
        
        # Split data into features (X) and target (y)
        X = data.drop('Class', axis=1)
        y = data['Class']
        
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        return X_train, X_test, y_train, y_test
    except Exception as e:
        print(f"Error preprocessing data: {e}")
        raise

def train_model(X_train, y_train):
    try:
        # Train a RandomForestClassifier model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Log the model with MLflow
        mlflow.sklearn.log_model(model, "random_forest")
        
        return model
    except Exception as e:
        print(f"Error training model: {e}")
        raise
