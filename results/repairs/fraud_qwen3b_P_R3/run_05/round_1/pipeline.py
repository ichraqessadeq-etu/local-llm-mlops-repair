import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn

def load_data(file_path):
    """Load data from a CSV file."""
    try:
        data = pd.read_csv(file_path)
        return data
    except FileNotFoundError:
        raise ValueError('File not found.')
    except Exception as e:
        raise Exception(f'An error occurred while loading data: {e}')

def preprocess_data(data):
    """Preprocess the data."""
    try:
        X = data.drop('Class', axis=1)
        y = data['Class']
        return (X, y)
    except Exception as e:
        raise Exception(f'An error occurred while preprocessing data: {e}')

def train_model(X_train, y_train):
    """Train a RandomForest model."""
    try:
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        return model
    except Exception as e:
        raise Exception(f'An error occurred while training the model: {e}')

def save_model(model, experiment_name, run_name):
    """Save the model using MLflow."""
    try:
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run(run_name=run_name):
            mlflow.sklearn.log_model(model, 'model')
            mlflow.log_metric('accuracy', accuracy_score(y_train, model.predict(X_train)))
        return mlflow.get_run_id()
    except Exception as e:
        raise Exception(f'An error occurred while saving the model: {e}')