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
        raise Exception(f"Failed to load data: {e}")

def preprocess_data(data):
    """Preprocess data by separating features and target."""
    try:
        X = data.drop('Class', axis=1)
        y = data['Class']
        return X, y
    except Exception as e:
        raise Exception(f"Failed to preprocess data: {e}")

def train_model(X_train, y_train):
    """Train a RandomForest model and log it using MLflow."""
    try:
        with mlflow.start_run():
            model = RandomForestClassifier()
            model.fit(X_train, y_train)
            mlflow.log_param("n_estimators", model.n_estimators)
            mlflow.log_param("max_depth", model.max_depth)
            mlflow.sklearn.log_model(model, "model")
            return model
    except Exception as e:
        raise Exception(f"Failed to train model: {e}")

def evaluate_model(model, X_test, y_test):
    """Evaluate the model and log metrics."""
    try:
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        classification_rep = classification_report(y_test, y_pred)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_text(classification_rep, "classification_report.txt")
        return accuracy, classification_rep
    except Exception as e:
        raise Exception(f"Failed to evaluate model: {e}")
