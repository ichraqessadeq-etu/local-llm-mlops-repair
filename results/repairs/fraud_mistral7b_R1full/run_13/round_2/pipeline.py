import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import mlflow
import mlflow.sklearn

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(data):
    # Perform necessary data preprocessing like filling missing values, encoding categorical variables, etc.
    # For simplicity, I'll assume no preprocessing is required.
    return data['Features'], data['Class']

def train_model(random_state=42):
    data = load_data('data/fraud.csv')
    X, y = preprocess_data(data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_state)

    with mlflow.start_run():
        model = RandomForestClassifier(random_state=random_state)
        model.fit(X_train, y_train)
        mlflow.sklearn.log_model(model, "random_forest")

        run_id = mlflow.get_run_id()
        print(f"Training run ID: {run_id}")
