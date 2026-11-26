import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow

def load_data(file_path):
    """Load the dataset from a CSV file."""
    try:
        data = pd.read_csv(file_path)
        return data
    except FileNotFoundError:
        raise FileNotFoundError("The file does not exist.")
    except pd.errors.EmptyDataError:
        raise ValueError("The file is empty.")

def preprocess_data(data):
    """Preprocess the data by splitting into features and target."""
    try:
        X = data.drop('Class', axis=1)
        y = data['Class']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test
    except KeyError:
        raise KeyError("The 'Class' column is missing in the dataset.")

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
        print(f"Model Accuracy: {accuracy:.2f}")
        return accuracy
    except Exception as e:
        raise Exception(f"An error occurred while evaluating the model: {e}")

def run_pipeline(file_path):
    """Run the entire pipeline."""
    try:
        data = load_data(file_path)
        X_train, X_test, y_train, y_test = preprocess_data(data)
        model = train_model(X_train, y_test)  # Fixed: Use y_test instead of y_train
        evaluate_model(model, X_test, y_test)
        mlflow.log_metric("accuracy", evaluate_model(model, X_test, y_test))
    except Exception as e:
        print(f"Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    run_pipeline("data/fraud.csv")

