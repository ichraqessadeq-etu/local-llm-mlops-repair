import pandas as pd
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
        X = data.drop(columns=['Class'])
        y = data['Class']
        return X, y
    except Exception as e:
        raise Exception(f"Failed to preprocess data: {e}")

def train_model(X_train, y_train):
    """Train a RandomForestClassifier model."""
    try:
        with mlflow.start_run():
            mlflow.log_param("model", "RandomForestClassifier")
            model = RandomForestClassifier(random_state=42)  # Added random_state
            model.fit(X_train, y_train)
            mlflow.sklearn.log_model(model, "model")
            y_pred = model.predict(X_train)
            accuracy = accuracy_score(y_train, y_pred)
            mlflow.log_metric("accuracy", accuracy)
            print(classification_report(y_train, y_pred))
            return model
    except Exception as e:
        raise Exception(f"Failed to train model: {e}")
