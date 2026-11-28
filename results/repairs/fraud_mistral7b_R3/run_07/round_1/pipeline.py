import mlflow
import pandas as pd
import sklearn.ensemble as rf
from sklearn.model_selection import train_test_split
from sklearn.exceptions import ConvergenceWarning

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(data):
    return (data, data['Amount'])

def train_model(X_train, y_train):
    with mlflow.start_run():
        try:
            model = rf.RandomForestClassifier(random_state=42)
            model.fit(X_train, y_train)
            mlflow.log_artifact(f'model_{mlflow.active_run().info.run_id}.pkl', artifact_location='models')
            return model
        except ConvergenceWarning:
            mlflow.log_warning('ConvergenceWarning during model training.')