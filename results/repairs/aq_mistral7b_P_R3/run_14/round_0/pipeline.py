import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import mlflow

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(data):
    X = data.drop('Anomaly', axis=1)
    y = data['Anomaly']
    return X, y

def train_model(X_train, y_train):
    with mlflow.start_run():
        mlflow.set_tag('target', 'anomaly_detector')
        mlflow.log_artifacts({'initial_data': data.to_csv()})

        model = RandomForestClassifier()
        model.fit(X_train, y_train)

        mlflow.sklearn.log_model(model, 'anomaly_detector')
