import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import mlflow
import mlflow.sklearn

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(data):
    return (data['X'], data['Class'])

def train_model(X_train, y_train):
    with mlflow.start_run():
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        mlflow.sklearn.log_model(model, 'random-forest')
        mlflow.log_param('n_estimators', model.n_estimators)
        mlflow.log_param('max_depth', model.max_depth)
        mlflow.log_param('min_samples_split', model.min_samples_split)
        mlflow.log_param('min_samples_leaf', model.min_samples_leaf)
        mlflow.log_param('random_state', model.random_state)
    return model