import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
from mlflow.exceptions import MlflowException

def load_data(file_path):
    """Load the dataset from the specified file path."""
    try:
        data = pd.read_csv(file_path)
        return data
    except FileNotFoundError:
        raise MlflowException(f'File not found at {file_path}')

def preprocess_data(data):
    """Preprocess the data by splitting it into features and target."""
    X = data.drop('Anomaly', axis=1)
    y = data['Anomaly']
    return (X, y)

def train_model(X_train, y_train):
    """Train a RandomForestClassifier model."""
    try:
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        return model
    except Exception as e:
        raise MlflowException(f'Error training model: {e}')

def evaluate_model(model, X_test, y_test):
    """Evaluate the model's performance."""
    try:
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        return accuracy
    except Exception as e:
        raise MlflowException(f'Error evaluating model: {e}')

def run_pipeline(file_path):
    """Run the entire pipeline from data loading to evaluation."""
    try:
        data = load_data(file_path)
        X, y = preprocess_data(data)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = train_model(X_train, y_train)
        accuracy = evaluate_model(model, X_test, y_test)
        print(f'Model accuracy: {accuracy:.2f}')
        return model
    except MlflowException as e:
        print(f'Pipeline error: {e}')
        return None