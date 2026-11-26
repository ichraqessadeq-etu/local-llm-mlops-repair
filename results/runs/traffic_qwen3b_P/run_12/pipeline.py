import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
from mlflow.exceptions import MlflowException

def load_data(file_path):
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        raise MlflowException(f"Error loading data: {e}")

def preprocess_data(data):
    try:
        # Define features and target
        features = ['temp', 'rain_1h', 'snow_1h', 'clouds_all', 'weather_main', 'IsHoliday', 'Hour', 'DayOfWeek', 'Month']
        target = 'Congestion'
        
        # Split data into features and target
        X = data[features]
        y = data[target]
        
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        return X_train, X_test, y_train, y_test
    except Exception as e:
        raise MlflowException(f"Error preprocessing data: {e}")

def train_model(X_train, y_train):
    try:
        # Initialize the RandomForestClassifier
        model = RandomForestClassifier(random_state=42)
        
        # Train the model
        model.fit(X_train, y_train)
        
        return model
    except Exception as e:
        raise MlflowException(f"Error training model: {e}")
