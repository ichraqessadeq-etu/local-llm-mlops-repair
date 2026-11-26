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
    except FileNotFoundError:
        raise ValueError("File not found")
    except pd.errors.EmptyDataError:
        raise ValueError("Empty data")

def preprocess_data(data):
    """Preprocess the data by splitting into features and target."""
    X = data.drop('Class', axis=1)
    y = data['Class']
    return X, y

def train_model(X_train, y_train):
    """Train a RandomForestClassifier."""
    try:
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        return model
    except Exception as e:
        mlflow.log_exception(e)
        raise ValueError("Model training failed")

def evaluate_model(model, X_test, y_test):
    """Evaluate the model."""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_text("classification_report", report)
    return accuracy, report

def run_pipeline(file_path):
    """Run the entire pipeline."""
    data = load_data(file_path)
    X, y = preprocess_data(data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = train_model(X_train, y_train)
    accuracy, report = evaluate_model(model, X_test, y_test)
    mlflow.log_metric("test_accuracy", accuracy)
    mlflow.log_text("test_classification_report", report)
    return model

