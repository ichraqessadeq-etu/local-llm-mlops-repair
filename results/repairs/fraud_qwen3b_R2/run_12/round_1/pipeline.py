import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
from mlflow.exceptions import MlflowException

def load_data(file_path):
    """Load data from a CSV file."""
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        raise MlflowException(f"Failed to load data: {e}")

def preprocess_data(data):
    """Preprocess the data."""
    # Example preprocessing: drop columns with missing values
    data = data.dropna()
    
    # Split features and target
    X = data.drop('Class', axis=1)
    y = data['Class']
    
    return X, y

def train_model(X_train, y_train):
    """Train a RandomForest model."""
    try:
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        return model
    except Exception as e:
        raise MlflowException(f"Failed to train model: {e}")

def evaluate_model(model, X_test, y_test):
    """Evaluate the model."""
    try:
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model accuracy: {accuracy}")
        return accuracy
    except Exception as e:
        raise MlflowException(f"Failed to evaluate model: {e}")
