import pandas as pd
import numpy as np
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
        raise FileNotFoundError(f"File {file_path} not found.")
    except pd.errors.EmptyDataError:
        raise pd.errors.EmptyDataError("The CSV file is empty.")

def preprocess_data(data):
    """Preprocess the data."""
    try:
        # Assume 'Class' is the target column
        X = data.drop('Class', axis=1)
        y = data['Class']
        
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        return X_train, y_train
    except KeyError:
        raise KeyError("The target column 'Class' is missing in the data.")

def train_model(X_train, y_train):
    """Train a RandomForest classifier."""
    try:
        # Initialize the model
        model = RandomForestClassifier(random_state=42)
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Log the model to MLflow
        mlflow.sklearn.log_model(model, "random_forest_model")
        
        return model
    except Exception as e:
        mlflow.log_metric("exception", str(e))
        raise Exception("Failed to train the model.")

def predict(model, X_test):
    """Predict using the trained model."""
    try:
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        mlflow.log_metric("accuracy", accuracy)
        return y_pred
    except Exception as e:
        mlflow.log_metric("exception", str(e))
        raise Exception("Failed to predict.")

def main():
    file_path = 'data/fraud.csv'
    try:
        data = load_data(file_path)
        X_train, y_train = preprocess_data(data)
        model = train_model(X_train, y_train)
        y_pred = predict(model, X_train)
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
