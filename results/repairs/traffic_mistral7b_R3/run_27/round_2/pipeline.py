import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from mlflow import log_artifact, start_run, end_run
Features = ['temp', 'rain_1h', 'snow_1h', 'clouds_all', 'weather_main', 'IsHoliday', 'Hour', 'DayOfWeek', 'Month']  # R3: features du schéma (toutes les colonnes sauf la cible)

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(data):
    return (data[Features], data['Congestion'])

def train_model(X_train, y_train):
    with start_run():
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        artifact_path = f'model'
        log_artifact(model, artifact_path)
        pass  # R3: API inexistante retirée
    return model