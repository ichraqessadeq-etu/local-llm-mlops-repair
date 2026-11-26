import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn

def load_data(file_path):
    """Load data from the specified CSV file."""
    try:
        data = pd.read_csv(file_path)
        return data
    except FileNotFoundError:
        raise FileNotFoundError(f"File {file_path} not found.")
    except pd.errors.EmptyDataError:
        raise pd.errors.EmptyDataError("The file is empty.")

def preprocess_data(data):
    """Preprocess the data by splitting into features and target, and splitting into training and testing sets."""
    try:
        X = data.drop('Congestion', axis=1)
        y = data['Congestion']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test
    except KeyError:
        raise KeyError("The 'Congestion' column is missing in the data.")

def train_model(X_train, y_train):
    """Train a RandomForestClassifier model."""
    try:
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        return model
    except Exception as e:
        raise Exception(f"An error occurred while training the model: {e}")

def evaluate_model(model, X_test, y_test):
    """Evaluate the model's performance."""
    try:
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model accuracy: {accuracy:.2f}")
    except Exception as e:
        raise Exception(f"An error occurred while evaluating the model: {e}")
