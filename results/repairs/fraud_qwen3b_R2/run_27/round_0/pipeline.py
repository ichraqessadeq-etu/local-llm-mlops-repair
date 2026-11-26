import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow

def load_data(file_path):
    """Load data from a CSV file."""
    try:
        data = pd.read_csv(file_path)
        return data
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return None

def preprocess_data(data):
    """Preprocess the data by splitting into features and target."""
    try:
        X = data.drop('Class', axis=1)
        y = data['Class']
        return X, y
    except KeyError:
        print("Error: 'Class' column not found in the data.")
        return None, None

def train_model(X_train, y_train):
    """Train a RandomForest model."""
    try:
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        return model
    except Exception as e:
        print(f"Error training the model: {e}")
        return None

def run_pipeline(file_path):
    """Run the entire pipeline."""
    data = load_data(file_path)
    if data is None:
        return

    X, y = preprocess_data(data)
    if X is None or y is None:
        return

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = train_model(X_train, y_train)
    if model is None:
        return

    # Log the model to MLflow
    mlflow.set_experiment("fraud_detection")
    mlflow.sklearn.log_model(model, "random_forest_model")

    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy:.2f}")
