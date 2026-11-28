import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import mlflow
_R3_FEATURES = ['temp', 'rain_1h', 'snow_1h', 'clouds_all', 'weather_main', 'IsHoliday', 'Hour', 'DayOfWeek', 'Month']  # R3: features du schéma

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(data):
    return (data[_R3_FEATURES], data['Congestion'])

def train_model(X_train, y_train):
    with mlflow.start_run():
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        mlflow.log_artifact(f'model', local_model=model)
        mlflow.log_param('num_trees', model.n_estimators)
        mlflow.log_metric('train_loss', model.fit_score(X_train, y_train))
    return model