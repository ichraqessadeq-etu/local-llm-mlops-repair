import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import mlflow
import mlflow.sklearn
import numpy as np

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(data, random_state=42):
    # Data preprocessing steps such as handling missing values, encoding categorical variables, etc.
    # For simplicity, I'll assume no preprocessing is needed.
    return data['Features'], data['Class']

def train_model(X_train, y_train, random_state=42):
    with mlflow.start_run():
        model = RandomForestClassifier(random_state=random_state)
        model.fit(X_train, y_train)
        mlflow.sklearn.log_model(model, "random-forest")
    return model
