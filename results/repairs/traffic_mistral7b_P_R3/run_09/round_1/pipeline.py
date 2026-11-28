import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from mlflow import pyfunc, tracking, sklearn

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(data):
    return (data, data['Congestion'])

def train_model(X_train, y_train):
    with mlflow.start_run():
        model = RandomForestClassifier(random_state=42)
        model = sklearn.sklearn_model_operator.fit(model, X_train, y_train)
        mlflow.sklearn.log_model(model, 'random-forest')
        mlflow.run.commit()
    return model