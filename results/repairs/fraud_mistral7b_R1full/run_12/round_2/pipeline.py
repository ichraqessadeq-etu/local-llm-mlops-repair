import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import mlflow

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(data):
    # Add any necessary preprocessing steps here
    return data.drop('Amount', axis=1), data['Amount']

def train_model(X_train, y_train, random_state=42):
    with mlflow.start_run():
        model = RandomForestClassifier(random_state=random_state)
        model.fit(X_train, y_train)
        mlflow.pyfunc.log_model(model, "model")
        mlflow.log_artifact(artifact_path="model", artifact_rel_path="models")
        track(f"Training loss: {model.loss_}")
        track(f"Training score: {model.score_}")
