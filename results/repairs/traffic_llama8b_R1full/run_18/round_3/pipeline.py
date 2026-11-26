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
        data.fillna(0, inplace=True)  # Replace NaN with 0

        # Convert categorical features to numerical
        data['weather_main'] = data['weather_main'].map({'Rain': 1, 'Snow': 2, 'Clouds': 3, 'Clear': 4, 'Thunderstorm': 5})
        data['Month'] = data['Month'].map({'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6, 'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12})

        # Scale features
        scaler = StandardScaler()
        X = scaler.fit_transform(data.drop('Congestion', axis=1))

        # Split data into features and target
        y = data['Congestion']

        # Check for NaN in X and y
        if np.isnan(X).any() or np.isnan(y).any():
            raise ValueError("Input contains NaN, infinity or a value too large for dtype('float32')")

        return X, y
    except Exception as e:
        print(f"Error preprocessing data: {e}")
        raise

def train_model(X_train, y_train):
    try:
        # Train RandomForest model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Log model to MLflow
        mlflow.sklearn.log_model(model, "random_forest_model")

        return model
    except Exception as e:
        print(f"Error training model: {e}")
        raise
