import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import mlflow
import logging
from sklearn import set_config
set_config(config_options={'display_params': False})

def load_data(file_path):
    """Load data from CSV file"""
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        logging.error(f'Error loading data: {e}')
        raise

def preprocess_data(data):
    """Preprocess data by encoding categorical variables and scaling numerical variables"""
    try:
        data['weather_main'] = pd.Categorical(data['weather_main']).codes
        data['Hour'] = pd.Categorical(data['Hour']).codes
        data['DayOfWeek'] = pd.Categorical(data['DayOfWeek']).codes
        data['Month'] = pd.Categorical(data['Month']).codes
        scaler = StandardScaler()
        data[['temp', 'rain_1h', 'snow_1h', 'clouds_all']] = scaler.fit_transform(data[['temp', 'rain_1h', 'snow_1h', 'clouds_all']])
        X = data.drop('Congestion', axis=1)
        y = data['Congestion']
        return (X, y)
    except Exception as e:
        logging.error(f'Error preprocessing data: {e}')
        raise

def train_model(X_train, y_train):
    """Train a RandomForest model"""
    try:
        mlflow.sklearn.log_model(RandomForestClassifier(n_estimators=100, random_state=42), 'random_forest_model')
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        mlflow.log_metric('accuracy', model.score(X_train, y_train))
        return model
    except Exception as e:
        logging.error(f'Error training model: {e}')
        raise
