import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.sklearn
import numpy as np

def load_data(file_path):
    data = pd.read_csv(file_path)
    data['weather_main'] = pd.Categorical(data['weather_main']).codes
    return data

def preprocess_data(data):
    return (data.drop('Congestion', axis=1), data['Congestion'])

def train_model(X_train, y_train):
    with mlflow.start_run():
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        mlflow.sklearn.log_model(model, 'random-forest')
    return model
