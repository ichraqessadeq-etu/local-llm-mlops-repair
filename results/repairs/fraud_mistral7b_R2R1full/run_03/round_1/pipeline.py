import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import mlflow
import mlflow.sklearn
from sklearn.utils.multiclass import type_of_target

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(data):
    return (data, data['Amount'])

def train_model(X_train, y_train):
    y_type = type_of_target(y_train)
    if y_type != 'continuous':
        with mlflow.start_run():
            model = RandomForestClassifier(random_state=42)
            model.fit(X_train, y_train)
            mlflow.sklearn.log_model(model, 'random-forest')
        run_id = mlflow.get_run_id()
        print(f'Trained model saved with run ID: {run_id}')
    else:
        raise ValueError("Continuous target not supported by RandomForestClassifier")
