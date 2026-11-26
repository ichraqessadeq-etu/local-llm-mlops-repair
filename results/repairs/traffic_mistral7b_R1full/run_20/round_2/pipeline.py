import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import mlflow
import mlflow.sklearn

# Define the Features list
Features = ['temp', 'rain_1h', 'snow_1h', 'clouds_all', 'weather_main', 'IsHoliday', 'Hour', 'DayOfWeek', 'Month']

def load_data(file_path):
    data = pd.read_csv(file_path)
    data.replace('Clouds', np.nan, inplace=True)  # Replace 'Clouds' with NaN
    return data

def preprocess_data(data):
    # Preprocessing steps like filling missing values, encoding categorical variables, etc.
    data[Features] = data[Features].astype(np.float32)  # Convert categorical columns to float32
    data = pd.get_dummies(data, columns=['weather_main', 'clouds_all'])  # One-hot encode categorical columns
    return data[Features], data['Congestion']

def train_model(X_train, y_train):
    with mlflow.start_run():
        model = RandomForestClassifier(random_state=42)  # Seed the random number generator
        model.fit(X_train, y_train)
        mlflow.sklearn.log_model(model, "random-forest")

    return model
