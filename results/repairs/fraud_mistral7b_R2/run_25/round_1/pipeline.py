import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import mlflow

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(data):
    return (data['Features'], data['Class'])

def train_model(X_train, y_train):
    with mlflow.start_run():
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        mlflow.log_artifact(f'model', model)
        mlflow.log_param('num_trees', model.n_estimators)
        mlflow.log_metric('train_loss', model.fit(X_train, y_train).score(X_train, y_train))
        mlflow.log_metric('val_loss', model.fit(X_train, y_train).score(X_val, y_val))
    return model