# pipeline.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import mlflow

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(data):
    data = data.replace({'weather_main': {'Rain': 0, 'Snow': 1, 'Clouds': 2}})
    return (data[['temp', 'rain_1h', 'snow_1h', 'clouds_all', 'IsHoliday', 'Hour', 'DayOfWeek', 'Month']], data['Congestion'])

def train_model(X_train, y_train):
    with mlflow.start_run():
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        mlflow.log_artifact('model', model)
        mlflow.log_param('num_trees', model.n_estimators)
        mlflow.log_metric('train_loss', model.fit(X_train, y_train).score(X_train, y_train))
        mlflow.log_metric('val_loss', model.fit(X_train, y_train).score(X_val, y_val))
    return model
