import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.sklearn

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(data):
    # Preprocessing steps such as filling missing values, encoding categorical variables, normalization, etc.
    # For simplicity, I'll assume no preprocessing is needed
    return data, data['Anomaly']

def train_model(X_train, y_train, random_state=42):
    with mlflow.start_run():
        model = RandomForestClassifier(random_state=random_state)
        model.fit(X_train, y_train)
        mlflow.sklearn.log_model(model, "anomaly_detector")

    return model
