import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.sklearn
import numpy as np

Features = ['PT08.S1(CO)', 'PT08.S2(NMHC)', 'PT08.S3(NOx)', 'PT08.S4(NO2)', 'PT08.S5(O3)', 'NOx(GT)', 'C6H6(GT)', 'T', 'RH', 'AH', 'Hour', 'DayOfWeek', 'Month']

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(data):
    # Add any necessary preprocessing steps here
    return data[Features], data['Anomaly']

def train_model(X_train, y_train):
    model = RandomForestClassifier(random_state=42)  # Seed the random number generator
    model.fit(X_train, y_train)
    mlflow.sklearn.log_model(model, "anomaly_detector")

    return model
