import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(data):
    features = data[['temp', 'rain_1h', 'snow_1h', 'clouds_all', 'weather_Main', 'IsHoliday', 'Hour', 'DayOfWeek', 'Month']]
    target = data['Congestion']
    return pd.concat([features, target], axis=1)

def train_model(X_train, y_train):
    with mlflow.start_run(run_name='RandomForest_Congestion_Detector'):
        try:
            model = RandomForestClassifier(random_state=42)
            model.fit(X_train, y_train)
            log_param('learning_rate', model.get_params()['max_depth'])
            return model
        except Exception as e:
            log_metric('training_error', 1)
            print(f'An error occurred during training: {e}')
