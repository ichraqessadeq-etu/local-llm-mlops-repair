import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import mlflow
import mlflow.sklearn
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

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
        imputer = SimpleImputer(strategy='mean')
        data[['temp', 'rain_1h', 'snow_1h', 'clouds_all', 'Hour', 'DayOfWeek', 'Month']] = imputer.fit_transform(data[['temp', 'rain_1h', 'snow_1h', 'clouds_all', 'Hour', 'DayOfWeek', 'Month']])

        # Convert categorical features to numerical
        encoder = OneHotEncoder(handle_unknown='ignore')
        encoded_data = encoder.fit_transform(data[['weather_main', 'Month']])
        encoded_data = pd.DataFrame(encoded_data.toarray())

        data = pd.concat([data, encoded_data], axis=1)
        data = data.drop(['weather_main', 'Month'], axis=1)

        # Scale features
        scaler = StandardScaler()
        X = scaler.fit_transform(data.drop('Congestion', axis=1))

        # Split data into features and target
        y = data['Congestion']

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
