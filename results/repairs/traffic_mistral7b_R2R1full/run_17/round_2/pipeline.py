import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import mlflow
import mlflow.sklearn

def load_data(file_path):
    return pd.read_csv(file_path, index_col=False)

def preprocess_data(data):
    data.columns = ['temp', 'rain_1h', 'snow_1h', 'clouds_all', 'weather_main', 'IsHoliday', 'Hour', 'DayOfWeek', 'Month']
    data = data.dropna()
    return (data[['temp', 'rain_1h', 'snow_1h', 'clouds_all', 'weather_main', 'IsHoliday', 'Hour', 'DayOfWeek', 'Month']], data['Congestion'])

def train_model(X_train, y_train):
    with mlflow.start_run():
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        mlflow.sklearn.log_model(model, 'random_forest')
    return model
