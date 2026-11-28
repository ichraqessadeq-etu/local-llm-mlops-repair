import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from mlflow import start_run, log_param, log_metric, log_artifact

def load_data(file_path):
    """Load data from a CSV file."""
    try:
        data = pd.read_csv(file_path)
        return data
    except FileNotFoundError:
        raise FileNotFoundError(f"File {file_path} not found.")
    except pd.errors.EmptyDataError:
        raise pd.errors.EmptyDataError("The file is empty.")

def preprocess_data(data):
    """Preprocess data by splitting into features and target."""
    try:
        X = data.drop('Class', axis=1)
        y = data['Class']
        return X, y
    except KeyError:
        raise KeyError("The target column 'Class' is missing.")

def train_model(X_train, y_train):
    """Train a RandomForestClassifier model."""
    try:
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        return model
    except Exception as e:
        raise Exception(f"Failed to train model: {e}")

def evaluate_model(model, X_test, y_test):
    """Evaluate the model's accuracy."""
    try:
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        return accuracy
    except Exception as e:
        raise Exception(f"Failed to evaluate model: {e}")

def run_pipeline(file_path):
    """Run the entire pipeline."""
    try:
        data = load_data(file_path)
        X, y = preprocess_data(data)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        with start_run():
            log_param("model", "RandomForestClassifier")
            log_param("random_state", 42)
            
            model = train_model(X_train, y_train)
            accuracy = evaluate_model(model, X_test, y_test)
            log_metric("accuracy", accuracy)
            
            log_artifact("model.pkl", "model.pkl")
            
        print(f"Model trained with accuracy: {accuracy}")
    except Exception as e:
        print(f"Pipeline failed: {e}")

if __name__ == "__main__":
    run_pipeline("data/fraud.csv")
