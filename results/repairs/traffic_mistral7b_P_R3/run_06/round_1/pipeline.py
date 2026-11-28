import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from mlflow import log_artifact, pyfunc

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(data):
    data = data.dropna()
    return (data.iloc[:, :-1], data.iloc[:, -1])

def train_model(X_train, y_train):
    model = RandomForestClassifier(random_state=42)
    with mlflow.start_run():
        model.fit(X_train, y_train)
        pyfunc.log_model(model, 'model', registered_model_name='congestion_detector')
        pass  # R3: API inexistante retirée
    return model