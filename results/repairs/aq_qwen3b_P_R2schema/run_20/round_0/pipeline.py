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
        raise ValueError("File not found.")
    except pd.errors.EmptyDataError:
        raise ValueError("The file is empty.")

def preprocess_data(data):
    """Preprocess the data."""
    # Assuming 'Anomaly' is the target column
    X = data.drop('Anomaly', axis=1)
    y = data['Anomaly']
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    """Train a RandomForest classifier."""
    try:
        with mlflow.start_run():
            # Log the model parameters
            mlflow.log_param("n_estimators", 100)
            mlflow.log_param("max_depth", 10)
            
            # Train the model
            model = RandomForestClassifier(n_estimators=100, max_depth=10)
            model.fit(X_train, y_train)
            
            # Log the model
            mlflow.sklearn.log_model(model, "anomaly_detector")
            
            # Evaluate the model
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            mlflow.log_metric("accuracy", accuracy)
            
            return model
    except Exception as e:
        mlflow.log_exception(e)
        raise e
