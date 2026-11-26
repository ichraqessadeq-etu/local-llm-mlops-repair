import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from mlflow import log_param, log_metric
from mlflow.sklearn import log_model

def load_data(file_path):
    """Load data from a CSV file."""
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        raise Exception(f'Failed to load data: {e}')

def preprocess_data(data):
    """Preprocess the data by separating features and target."""
    try:
        X = data.drop('Class', axis=1)
        y = data['Class']
        return (X, y)
    except Exception as e:
        raise Exception(f'Failed to preprocess data: {e}')

def train_model(X_train, y_train):
    """Train a RandomForestClassifier and log metrics using MLflow."""
    try:
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        log_param('n_estimators', model.n_estimators)
        log_param('max_depth', model.max_depth)
        y_pred = model.predict(X_train)
        accuracy = accuracy_score(y_train, y_pred)
        log_metric('accuracy', accuracy)
        log_model(model, 'random_forest_model')
        return model
    except Exception as e:
        raise Exception(f'Failed to train model: {e}')