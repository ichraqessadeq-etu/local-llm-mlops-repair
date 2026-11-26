import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from mlflow import log_artifact, pyfunc, track, set_tracking_params

set_tracking_params(experiment_name="AirQualityAnomalyDetector")

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(data):
    # Preprocessing steps like filling missing values, encoding categorical variables, etc.
    # For simplicity, I'm assuming no missing values and no encoding needed.
    return data.drop('Anomaly', axis=1), data['Anomaly']

def train_model(X_train, y_train, random_state=42):
    with mlflow.start_run():
        model = RandomForestClassifier(random_state=random_state)
        model.fit(X_train, y_train)
        mlflow.pyfunc.log_model(model, "model")
        print(f"Model trained and logged with MLflow run ID: {mlflow.get_run_id()}")
