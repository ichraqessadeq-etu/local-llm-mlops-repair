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
        raise ValueError('File not found')
    except pd.errors.EmptyDataError:
        raise ValueError('Empty data in the file')

def preprocess_data(data):
    """Preprocess the data by splitting it into features and target, and handling missing values."""
    try:
        X = data.drop(columns=['Anomaly'])
        y = data['Anomaly']
        X = X.fillna(X.mean())
        return (X, y)
    except KeyError:
        raise ValueError('Missing column in the data')

def train_model(X, y):
    """Train a RandomForest classifier and log the model to MLflow."""
    try:
        with mlflow.start_run():
            model = RandomForestClassifier(random_state=42)
            model.fit(X, y)
            mlflow.sklearn.log_model(model, 'random_forest_model')
            return model
    except Exception as e:
        mlflow.log_exception(e)
        raise

def evaluate_model(model, X_test, y_test):
    """Evaluate the model's performance."""
    try:
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        mlflow.log_metric('accuracy', accuracy)
        return accuracy
    except Exception as e:
        mlflow.log_exception(e)
        raise