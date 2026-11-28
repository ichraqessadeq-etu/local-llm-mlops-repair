import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from mlflow import log_artifact, pyfunc
_R3_FEATURES = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']  # R3: features du schéma

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(data):
    return (data[_R3_FEATURES], data['Class'])

def train_model(X_train, y_train):
    with mlflow.start_run():
        pass  # R3: API inexistante retirée
        pass  # R3: API inexistante retirée
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        mlflow.pyfunc.log_model(pyfunc.DistributedModel.from_flask(model), artifact_path='model')
        print(f'Model with version {mlflow.get_run_id()} has been saved.')