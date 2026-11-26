import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import mlflow
import numpy as np

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(data):
    # Data preprocessing steps like filling NaN, encoding categorical variables, etc.
    # For simplicity, I'm assuming that the data is already preprocessed.
    return data['temp'], data['rain_1h'], data['snow_1h'], data['clouds_all'], data['weather_main'], data['IsHoliday'], data['Hour'], data['DayOfWeek'], data['Month'], data['Congestion']

def train_model(X_train, y_train):
    with mlflow.start_run():
        model = RandomForestClassifier(random_state=42)  # Seed the random forest to ensure reproducibility
        model.fit(X_train, y_train)
        mlflow.log_artifact(f"model", local_model=model)
        mlflow.log_param("num_trees", model.n_estimators)
        mlflow.log_metric("train_loss", model.fit(X_train, y_train).score(X_train, y_train))
        mlflow.log_metric("val_loss", model.fit(X_train, y_train).score(X_val, y_val))

    return model
