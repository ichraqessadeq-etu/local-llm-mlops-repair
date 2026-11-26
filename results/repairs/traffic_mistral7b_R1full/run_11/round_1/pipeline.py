import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from mlflow.pyfunc import local_mode, DynamicArtifact
from mlflow import log_artifact, start_run, end_run

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(data):
    # Add any preprocessing steps here, such as normalization, encoding, etc.
    return np.array(data.drop('Congestion', axis=1)), data['Congestion']

def train_model(X_train, y_train):
    with start_run(run_name='RandomForestCongestionDetector', experiment_id='0'):
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        artifact_path = DynamicArtifact('model')
        artifact_path.set_local_path(f'./models/model.pkl')
        log_artifact(artifact_path)
    return model
