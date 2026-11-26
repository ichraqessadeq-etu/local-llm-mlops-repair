import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.sklearn
import numpy as np

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(data, random_state=42):
    # Perform necessary preprocessing like filling missing values, encoding categorical variables, etc.
    # For simplicity, let's assume the data is already preprocessed
    data['Features'] = data[['temp', 'rain_1h', 'snow_1h', 'clouds_all', 'weather_Main', 'IsHoliday', 'Hour', 'DayOfWeek', 'Month']]
    data['Congestion'] = data['Congestion'].astype(int)
    data = data.dropna()
    data['weather_Main'].fillna('', inplace=True)
    X, y = train_test_split(data[['Features']], data['Congestion'], test_size=0.2, random_state=random_state)
    return X, y

def train_model(X_train, y_train, random_state=42):
    with mlflow.start_run():
        model = RandomForestClassifier(random_state=random_state)
        model.fit(X_train, y_train)
        mlflow.sklearn.log_model(model, "random-forest")

    return model
