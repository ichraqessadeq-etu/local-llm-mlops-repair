import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import mlflow

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(data):
    # Perform data preprocessing like feature scaling, handling missing values, etc.
    # Return preprocessed X and y

def train_model(X_train, y_train):
    with mlflow.start_run():
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        mlflow.log_artifact(f"model", model)
