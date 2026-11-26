import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import mlflow
import os

Features = ['PT08.S1(CO)', 'PT08.S2(NMHC)', 'PT08.S3(NOx)', 'PT08.S4(NO2)', 'PT08.S5(O3)', 'NOx(GT)', 'C6H6(GT)', 'T', 'RH', 'AH', 'Hour', 'DayOfWeek', 'Month']

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(data):
    return (data[Features], data['Anomaly'])

def train_model(X_train, y_train):
    with mlflow.start_run():
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        model_path = os.path.join('models', f'random_forest_{mlflow.active_run().info.run_id}.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        return model
