import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import mlflow
import mlflow.sklearn

def load_data(file_path):
    """Load data from a CSV file."""
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        raise ValueError(f'Error loading data: {e}')

def preprocess_data(data):
    """Preprocess data by separating features and target."""
    try:
        X = data.drop(columns=['Class'])
        y = data['Class']
        return (X, y)
    except Exception as e:
        raise ValueError(f'Error preprocessing data: {e}')

def train_model(X_train, y_train):
    """Train a RandomForestClassifier."""
    try:
        with mlflow.start_run():
            model = RandomForestClassifier(random_state=42)
            model.fit(X_train, y_train)
            mlflow.log_param('n_estimators', model.n_estimators)
            mlflow.log_param('max_depth', model.max_depth)
            mlflow.sklearn.log_model(model, 'model')
            return model
    except Exception as e:
        raise ValueError(f'Error training model: {e}')

def evaluate_model(model, X_test, y_test):
    """Evaluate the model."""
    try:
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        mlflow.log_metric('accuracy', accuracy)
        mlflow.log_text(report, 'classification_report.txt')
        return (accuracy, report)
    except Exception as e:
        raise ValueError(f'Error evaluating model: {e}')