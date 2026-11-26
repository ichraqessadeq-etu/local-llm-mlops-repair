import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn

def load_data(file_path):
    """Load the dataset from a CSV file."""
    try:
        data = pd.read_csv(file_path)
        return data
    except FileNotFoundError:
        raise ValueError("The file does not exist.")
    except pd.errors.EmptyDataError:
        raise ValueError("The file is empty.")
    except pd.errors.ParserError:
        raise ValueError("The file is not a valid CSV.")

def preprocess_data(data):
    """Preprocess the data by splitting it into features and target."""
    try:
        X = data.drop('Class', axis=1)
        y = data['Class']
        return X, y
    except KeyError:
        raise ValueError("The 'Class' column is missing.")

def train_model(X_train, y_train):
    """Train a RandomForestClassifier model."""
    try:
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        return model
    except Exception as e:
        raise ValueError(f"An error occurred while training the model: {e}")

def evaluate_model(model, X_test, y_test):
    """Evaluate the model's performance."""
    try:
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        return accuracy
    except Exception as e:
        raise ValueError(f"An error occurred while evaluating the model: {e}")
