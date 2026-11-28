import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from mlflow import log_metric, log_param, log_artifact
import mlflow

def load_data(file_path):
    """Load the dataset from the specified file path."""
    try:
        data = pd.read_csv(file_path)
        return data
    except FileNotFoundError:
        raise ValueError("File not found: {}".format(file_path))
    except pd.errors.EmptyDataError:
        raise ValueError("Empty file: {}".format(file_path))

def preprocess_data(data):
    """Preprocess the data by splitting into features and target."""
    X = data.drop('Class', axis=1)
    y = data['Class']
    return X, y

def train_model(X_train, y_train):
    """Train a RandomForest classifier."""
    try:
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        return model
    except Exception as e:
        raise ValueError("Failed to train model: {}".format(e))

def evaluate_model(model, X_test, y_test):
    """Evaluate the model's performance."""
    try:
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        log_metric("accuracy", accuracy)
        print(f"Accuracy: {accuracy}")
        return accuracy
    except Exception as e:
        raise ValueError("Failed to evaluate model: {}".format(e))
