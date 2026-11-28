import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from mlflow import log_artifact, pyfunc
features = ['PT08.S1(CO)', 'PT08.S2(NMHC)', 'PT08.S3(NOx)', 'PT08.S4(NO2)', 'PT08.S5(O3)', 'NOx(GT)', 'C6H6(GT)', 'T', 'RH', 'AH', 'Hour', 'DayOfWeek', 'Month']  # R3: features du schéma (toutes les colonnes sauf la cible)

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(data):
    return (data[features], data['Anomaly'])

def train_model(X_train, y_train):
    model = RandomForestClassifier(random_state=42)
    pass  # R3: API inexistante retirée
    model.fit(X_train, y_train)
    return model

def save_model(model, experiment_name):
    mlflow.set_tracking_artifact(model, artifact_path=f'models/{experiment_name}')