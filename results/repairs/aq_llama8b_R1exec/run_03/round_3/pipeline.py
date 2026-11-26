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
        # Handle missing values
        data.fillna(data.mean(), inplace=True)
        
        # One-hot encoding for categorical features
        categorical_cols = ['DayOfWeek', 'Month']
        data = pd.get_dummies(data, columns=categorical_cols)
        
        # Select numerical columns
        numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns
        
        # Scale numerical features
        scaler = StandardScaler()
        data[numerical_cols] = scaler.fit_transform(data[numerical_cols])
        
        # Split data into features (X) and target (y)
        X = data.drop(['Anomaly'], axis=1)
        y = data['Anomaly']
        
        return X, y
    except Exception as e:
        print(f"Error preprocessing data: {e}")
        raise

def train_model(X_train, y_train):
    try:
        # Train a RandomForest classifier
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Log the model with MLflow
        mlflow.sklearn.log_model(model, "random_forest")
        
        return model
    except Exception as e:
        print(f"Error training model: {e}")
        raise
