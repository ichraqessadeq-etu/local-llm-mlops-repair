import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import mlflow
import mlflow.sklearn

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(data):
    return (data['temp'], data['rain_1h'], data['snow_1h'], data['clouds_all'], data['weather_main'], data['IsHoliday'], data['Hour'], data['DayOfWeek'], data['Month'], data['Congestion'])

def train_model(X_train, y_train):
    with mlflow.start_run():
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        mlflow.sklearn.log_model(model, 'random-forest')