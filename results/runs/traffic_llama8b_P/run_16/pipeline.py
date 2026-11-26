import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        mlflow.log_error(f"Error loading data: {str(e)}")
        raise

def preprocess_data(data):
    try:
        # Convert categorical variables to numerical
        data['weather_main'] = data['weather_main'].map({'Clear': 0, 'Clouds': 1, 'Rain': 2, 'Snow': 3})
        data['Holiday'] = data['Holiday'].map({False: 0, True: 1})
        
        # Scale numerical features
        scaler = StandardScaler()
        data[['temp', 'rain_1h', 'snow_1h', 'clouds_all']] = scaler.fit_transform(data[['temp', 'rain_1h', 'snow_1h', 'clouds_all']])
        
        # Split data into features (X) and target (y)
        X = data.drop('Congestion', axis=1)
        y = data['Congestion']
        
        return X, y
    except Exception as e:
        mlflow.log_error(f"Error preprocessing data: {str(e)}")
        raise

def train_model(X_train, y_train):
    try:
        # Train a RandomForest classifier
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Log the model
        mlflow.sklearn.log_model(model, "random_forest")
        
        return model
    except Exception as e:
        mlflow.log_error(f"Error training model: {str(e)}")
        raise
