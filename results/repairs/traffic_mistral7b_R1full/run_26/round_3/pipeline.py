import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn

Features = ['temp', 'rain_1h', 'snow_1h', 'clouds_all', 'weather_main', 'IsHoliday', 'Hour', 'DayOfWeek', 'Month']

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(data):
    # Preprocessing steps like filling missing values, encoding categorical variables, etc.
    data[Features] = data[Features].astype(np.float32)
    data['clouds_all'] = data['clouds_all'].astype('category')
    data['weather_main'] = data['weather_main'].astype('category')
    return data[Features], data['Congestion']

def train_model(X_train, y_train):
    np.random.seed(42)  # Set a global seed for reproducibility
    with mlflow.start_run():
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        mlflow.sklearn.log_model(model, "random-forest")

    return model
