import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import mlflow
import logging

logging.basicConfig(level=logging.INFO)

def load_data(file_path):
    """Load data from CSV file"""
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise

def preprocess_data(data):
    """Preprocess data by encoding categorical variables and scaling numerical variables"""
    try:
        # Encode categorical variables
        data['weather_main'] = pd.Categorical(data['weather_main']).codes
        data['Hour'] = pd.Categorical(data['Hour']).codes
        data['DayOfWeek'] = pd.Categorical(data['DayOfWeek']).codes
        data['Month'] = pd.Categorical(data['Month']).codes

        # Scale numerical variables
        scaler = StandardScaler()
        data[['temp', 'rain_1h', 'snow_1h', 'clouds_all']] = scaler.fit_transform(data[['temp', 'rain_1h', 'snow_1h', 'clouds_all']])

        # Split data into features (X) and target (y)
        X = data.drop('Congestion', axis=1)
        y = data['Congestion']

        return X, y
    except Exception as e:
        logging.error(f"Error preprocessing data: {e}")
        raise

def train_model(X_train, y_train):
    """Train a RandomForest model"""
    try:
        mlflow.sklearn.log_params(RandomForestClassifier(n_estimators=100))
        model = RandomForestClassifier(n_estimators=100)
        model.fit(X_train, y_train)
        mlflow.sklearn.log_model(model, "random_forest_model")
        return model
    except Exception as e:
        logging.error(f"Error training model: {e}")
        raise
