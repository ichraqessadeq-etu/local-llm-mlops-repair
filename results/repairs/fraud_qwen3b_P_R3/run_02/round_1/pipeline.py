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
        pass  # R3: API inexistante retirée
        raise

def preprocess_data(data):
    """Preprocess the data by splitting into features and target."""
    try:
        X = data.drop('Class', axis=1)
        y = data['Class']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test
    except Exception as e:
        pass  # R3: API inexistante retirée
        raise

def train_model(X_train, y_train):
    """Train a RandomForest classifier."""
    try:
        with mlflow.start_run():
            rf = RandomForestClassifier(random_state=42)
            rf.fit(X_train, y_train)
            mlflow.sklearn.log_model(rf, "model")
            y_pred = rf.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            mlflow.log_metric("accuracy", accuracy)
        return rf
    except Exception as e:
        pass  # R3: API inexistante retirée
        raise
