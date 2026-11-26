import pandas as pd
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
        # Convert categorical variables to numerical
        data['weather_main'] = data['weather_main'].astype('category').cat.codes
        data['DayOfWeek'] = data['DayOfWeek'].astype('category').cat.codes
        data['Month'] = data['Month'].astype('category').cat.codes

        # Scale features
        scaler = StandardScaler()
        X = scaler.fit_transform(data.drop('Congestion', axis=1))

        # Split data into features (X) and target (y)
        y = data['Congestion']
        return X, y
    except Exception as e:
        logging.error(f"Error preprocessing data: {e}")
        raise

def train_model(X_train, y_train):
    try:
        # Train RandomForest model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Log model with MLflow
        mlflow.sklearn.log_model(model, "random_forest")

        return model
    except Exception as e:
        logging.error(f"Error training model: {e}")
        raise
