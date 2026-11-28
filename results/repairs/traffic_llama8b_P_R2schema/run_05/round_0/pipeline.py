import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import mlflow
import mlflow.sklearn

def load_data(file_path):
    """Loads data from CSV file"""
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        mlflow.log_error(f"Error loading data: {e}")
        raise

def preprocess_data(data):
    """Preprocesses data by encoding categorical variables and scaling numerical variables"""
    try:
        X = data.drop(['Congestion'], axis=1)
        y = data['Congestion']
        X = pd.get_dummies(X, columns=['weather_main', 'IsHoliday'])
        scaler = StandardScaler()
        X[['temp', 'rain_1h', 'snow_1h', 'clouds_all']] = scaler.fit_transform(X[['temp', 'rain_1h', 'snow_1h', 'clouds_all']])
        return X, y
    except Exception as e:
        mlflow.log_error(f"Error preprocessing data: {e}")
        raise

def train_model(X_train, y_train):
    """Trains a RandomForestClassifier model"""
    try:
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        mlflow.sklearn.log_model(model, "random_forest")
        return model
    except Exception as e:
        mlflow.log_error(f"Error training model: {e}")
        raise
