import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import mlflow
import os

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(data):
    # Add any necessary preprocessing steps here, such as normalization or feature engineering
    X = data.drop('Class', axis=1)
    y = data['Class']
    return X, y

def train_model(X_train, y_train):
    random_state = np.random.randint(0, 100)
    with mlflow.start_run():
        model = RandomForestClassifier(random_state=random_state)
        model.fit(X_train, y_train)
        mlflow.sklearn.log_model(model, "random-forest")
