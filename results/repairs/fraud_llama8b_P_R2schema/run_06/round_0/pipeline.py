import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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
        
        # Scale features
        scaler = StandardScaler()
        X = scaler.fit_transform(data.drop('Class', axis=1))
        
        # Encode target variable
        y = data['Class'].map({'0': 0, '1': 1})
        
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        return X_train, X_test, y_train, y_test
    except Exception as e:
        print(f"Error preprocessing data: {e}")
        raise

def train_model(X_train, y_train):
    try:
        # Train a random forest classifier
        with mlflow.start_run():
            mlflow.set_experiment('fraud_detection')
            mlflow.log_param('model_type', 'RandomForestClassifier')
            mlflow.log_param('n_estimators', 100)
            mlflow.log_param('max_depth', 10)
            
            model = RandomForestClassifier(n_estimators=100, max_depth=10)
            model.fit(X_train, y_train)
            
            mlflow.sklearn.log_model(model, 'model')
            
            return model
    except Exception as e:
        print(f"Error training model: {e}")
        raise
