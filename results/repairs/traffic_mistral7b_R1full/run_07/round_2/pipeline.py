import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.preprocessing import LabelEncoder

def load_data(file_path):
    data = pd.read_csv(file_path)
    # Encode categorical features
    label_encoder = LabelEncoder()
    data['weather_main'] = label_encoder.fit_transform(data['weather_main'])
    return data

def preprocess_data(data):
    # Preprocessing steps like filling missing values, encoding categorical variables, etc.
    # For simplicity, I'm assuming no missing values
    X = data[['temp', 'rain_1h', 'snow_1h', 'clouds_all', 'weather_main', 'IsHoliday', 'Hour', 'DayOfWeek', 'Month']]
    y = data['Congestion']
    return X, y

def train_model(X_train, y_train):
    with mlflow.start_run():
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        mlflow.sklearn.log_model(model, "random-forest")
