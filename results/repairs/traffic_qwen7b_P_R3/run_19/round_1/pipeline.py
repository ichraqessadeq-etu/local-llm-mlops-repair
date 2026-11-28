# pipeline.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn

def load_data(file_path):
    """
    Load data from a CSV file.
    
    Parameters:
    - file_path: str, path to the CSV file.
    
    Returns:
    - df: pandas DataFrame containing the data.
    """
    df = pd.read_csv(file_path)
    return df

def preprocess_data(data):
    """
    Preprocess the data by separating features and target.
    
    Parameters:
    - data: pandas DataFrame containing the data.
    
    Returns:
    - X: pandas DataFrame containing the features.
    - y: pandas Series containing the target.
    """
    X = data.drop(columns=['Congestion'])
    y = data['Congestion']
    return X, y

def train_model(X_train, y_train):
    """
    Train a RandomForestClassifier model.
    
    Parameters:
    - X_train: pandas DataFrame containing the training features.
    - y_train: pandas Series containing the training target.
    
    Returns:
    - model: trained RandomForestClassifier model.
    """
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    with mlflow.start_run():
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("random_state", 42)
        model.fit(X_train, y_train)
        mlflow.sklearn.log_model(model, "model")
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model on the test set.
    
    Parameters:
    - model: trained model.
    - X_test: pandas DataFrame containing the test features.
    - y_test: pandas Series containing the test target.
    
    Returns:
    - accuracy: float, accuracy score of the model on the test set.
    """
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy
