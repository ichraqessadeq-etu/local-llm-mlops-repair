import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        mlflow.set_context("failed")
        mlflow.log_param("error", str(e))
        raise

def preprocess_data(data):
    try:
        X = data.drop(['Congestion'], axis=1)
        y = data['Congestion']
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        return train_test_split(X, y, test_size=0.2, random_state=42)
    except Exception as e:
        mlflow.set_context("failed")
        mlflow.log_param("error", str(e))
        raise

def train_model(X_train, y_train):
    try:
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        mlflow.sklearn.log_model(model, "random_forest")
        return model
    except Exception as e:
        mlflow.set_context("failed")
        mlflow.log_param("error", str(e))
        raise
