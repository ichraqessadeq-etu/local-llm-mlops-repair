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
    """Preprocess the data by separating features and target."""
    try:
        X = data.drop(columns=['Class'])
        y = data['Class']
        return X, y
    except Exception as e:
        raise Exception(f"Failed to preprocess data: {e}")

def train_model(X_train, y_train):
    """Train a RandomForest model."""
    try:
        with mlflow.start_run():
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            mlflow.log_param("n_estimators", 100)
            mlflow.log_param("random_state", 42)
            mlflow.sklearn.log_model(model, "model")
            y_pred = model.predict(X_train)
            accuracy = accuracy_score(y_train, y_pred)
            mlflow.log_metric("accuracy", accuracy)
            print(f"Model trained with accuracy: {accuracy}")
            return model
    except Exception as e:
        raise Exception(f"Failed to train model: {e}")

def evaluate_model(model, X_test, y_test):
    """Evaluate the trained model."""
    try:
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        print(f"Model evaluation: Accuracy = {accuracy}")
        print(f"Classification Report:\n{report}")
    except Exception as e:
        raise Exception(f"Failed to evaluate model: {e}")
