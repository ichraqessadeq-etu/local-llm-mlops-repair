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
    except Exception as e:
        raise ValueError(f'Failed to load data: {e}')

def preprocess_data(data):
    """Preprocess the data."""
    try:
        X = data.drop(columns=['Congestion'])
        y = data['Congestion']
        return (X, y)
    except Exception as e:
        raise ValueError(f'Failed to preprocess data: {e}')

def train_model(X_train, y_train):
    """Train a RandomForest model."""
    try:
        with mlflow.start_run():
            model = RandomForestClassifier(random_state=42)
            model.fit(X_train, y_train)
            mlflow.log_param('n_estimators', 100)
            mlflow.log_param('max_depth', 10)
            mlflow.sklearn.log_model(model, 'model')
            y_pred = model.predict(X_train)
            accuracy = accuracy_score(y_train, y_pred)
            mlflow.log_metric('train_accuracy', accuracy)
            return model
    except Exception as e:
        raise ValueError(f'Failed to train model: {e}')