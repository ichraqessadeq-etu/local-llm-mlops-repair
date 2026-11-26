import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import mlflow
import mlflow.sklearn
import os

def load_data(file_path):
    """
    Load data from a CSV file.
    
    Args:
    file_path (str): Path to the CSV file.
    
    Returns:
    pd.DataFrame: The loaded data.
    """
    return pd.read_csv(file_path)

def preprocess_data(data):
    """
    Preprocess the data by separating features and target.
    
    Args:
    data (pd.DataFrame): The raw data.
    
    Returns:
    tuple: A tuple containing the features (X) and target (y).
    """
    X = data.drop(columns=['Class'])
    y = data['Class']
    return X, y

def train_model(X_train, y_train):
    """
    Train a RandomForestClassifier model.
    
    Args:
    X_train (pd.DataFrame): The training features.
    y_train (pd.Series): The training target.
    
    Returns:
    RandomForestClassifier: The trained model.
    """
    with mlflow.start_run():
        mlflow.log_param("model", "RandomForestClassifier")
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        mlflow.sklearn.log_model(model, "model")
        y_pred = model.predict(X_train)
        accuracy = accuracy_score(y_train, y_pred)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_text(classification_report(y_train, y_pred), "classification_report.txt")
        return model

def main():
    file_path = 'data/fraud.csv'
    data = load_data(file_path)
    X, y = preprocess_data(data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = train_model(X_train, y_train)
    print(f"Model trained with accuracy: {accuracy_score(y_train, model.predict(X_train)):.4f}")

if __name__ == "__main__":
    main()
