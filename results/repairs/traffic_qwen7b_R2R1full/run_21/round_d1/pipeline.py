import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from mlflow import log_param, log_metric
from mlflow.sklearn import log_model

def load_data(file_path):
    """Load data from a CSV file."""
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        raise Exception(f"Error loading data: {e}")

def preprocess_data(data):
    """Preprocess data for model training."""
    try:
        # Select features and target
        X = data[Features]
        y = data['Congestion']
        
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        return X_train, X_test, y_train, y_test
    except Exception as e:
        raise Exception(f"Error preprocessing data: {e}")

def train_model(X_train, y_train):
    """Train a RandomForest model."""
    try:
        # Initialize and train the model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Log model parameters and metrics
        log_param("n_estimators", 100)
        y_pred = model.predict(X_train)
        log_metric("accuracy", accuracy_score(y_train, y_pred))
        
        # Log the model
        log_model(model, "random_forest_model")
        
        return model
    except Exception as e:
        raise Exception(f"Error training model: {e}")

# Example usage
if __name__ == "__main__":
    data = load_data("data/traffic.csv")
    X_train, X_test, y_train, y_test = preprocess_data(data)
    model = train_model(X_train, y_train)
