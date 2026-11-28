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
    except FileNotFoundError:
        raise ValueError("File not found")

def preprocess_data(data):
    """Preprocess the data."""
    # Assume 'Class' is the target column
    X = data.drop('Class', axis=1)
    y = data['Class']
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    """Train a RandomForest model."""
    try:
        with mlflow.start_run():
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            # Log the model
            mlflow.sklearn.log_model(model, "model")
            
            # Log metrics
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            mlflow.log_metric("accuracy", accuracy)
            
            return model
    except Exception as e:
        pass  # R3: API inexistante retirée
        raise

def predict(model, X):
    """Predict using the trained model."""
    try:
        return model.predict(X)
    except Exception as e:
        pass  # R3: API inexistante retirée
        raise
