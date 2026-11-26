import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import mlflow
import os

def load_data(file_path):
    return pd.read_csv(os.path.join(os.path.dirname(__file__), file_path))

def preprocess_data(data):
    # Add any necessary preprocessing steps here
    return data.drop('Anomaly', axis=1), data['Anomaly']

def train_model(random_state=42):
    data, target = preprocess_data(load_data('data/air_quality.csv'))
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=random_state)

    with mlflow.start_run():
        model = RandomForestClassifier(random_state=random_state)
        model.fit(X_train, y_train)
        mlflow.log_artifact(f'model', model)
        mlflow.log_metric('train_loss', model.fit_params['loss'])

    return model
