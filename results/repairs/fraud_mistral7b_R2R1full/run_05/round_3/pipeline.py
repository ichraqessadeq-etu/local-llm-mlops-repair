import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import mlflow

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(data):
    data = data.drop('Amount', axis=1)
    return (data, data['Amount'])

def train_model(X_train, y_train, Amount):
    model = RandomForestClassifier(random_state=42)
    mlflow_model = mlflow.sklearn.pyfunc.sklearn_model(model)
    mlflow.sklearn.autolog()
    mlflow.sklearn.log_model(mlflow_model, "model")
    return mlflow_model
