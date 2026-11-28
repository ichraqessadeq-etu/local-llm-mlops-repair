import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from mlflow import log_artifact, pyfunc, traceback

def load_data(file_path):
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        log_artifact(f'load_data_error.txt', 'text/plain')
        open('load_data_error.txt', 'w').write(traceback.format_exception(etype=e, value=e, tb=e.__traceback__))
        raise e

def preprocess_data(data):
    data.fillna(0, inplace=True)
    X = data.drop('Anomaly', axis=1)
    y = data['Anomaly']
    return (X, y)

def train_model(X_train, y_train):
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    mlflow_model = pyfunc.PyFuncModel.from_sklearn(model)
    mlflow.sklearn.log_model(mlflow_model, 'model')