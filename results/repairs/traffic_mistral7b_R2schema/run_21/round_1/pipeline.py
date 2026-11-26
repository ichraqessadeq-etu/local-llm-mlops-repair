import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from mlflow import trace_experiment, log_param, log_artifact, start_run, log_metric

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(data):
    return (data['Features'], data['Congestion'])

def train_model(X_train, y_train):
    with mlflow.start_run(run_name='RandomForest_Congestion_Detector'):
        try:
            model = RandomForestClassifier(random_state=42)
            model.fit(X_train, y_train)
            log_param('learning_rate', model.get_params()['max_depth'])
            log_artifact('model.pkl', model)
            return model
        except Exception as e:
            log_metric('training_error', 1)
            print(f'An error occurred during training: {e}')