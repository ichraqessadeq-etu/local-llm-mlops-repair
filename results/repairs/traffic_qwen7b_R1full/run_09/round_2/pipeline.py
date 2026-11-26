import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn

def load_data(file_path):
    """
    Load data from a CSV file.
    
    Args:
        file_path (str): Path to the CSV file.
    
    Returns:
        pd.DataFrame: Loaded data.
    """
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        raise Exception(f"Failed to load data: {e}")

def preprocess_data(data):
    """
    Preprocess the data by splitting into features (X) and target (y).
    
    Args:
        data (pd.DataFrame): Input data.
    
    Returns:
        tuple: Features (X) and target (y).
    """
    try:
        # Convert non-numeric columns to numeric, coercing errors to NaN
        data = data.apply(pd.to_numeric, errors='coerce')
        
        # Drop rows with any NaN values
        data = data.dropna()
        
        X = data.drop(columns=['Congestion'])
        y = data['Congestion']
        return X, y
    except Exception as e:
        raise Exception(f"Failed to preprocess data: {e}")

def train_model(X_train, y_train):
    """
    Train a RandomForestClassifier model.
    
    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.
    
    Returns:
        RandomForestClassifier: Trained model.
    """
    try:
        with mlflow.start_run():
            mlflow.log_param("model", "RandomForestClassifier")
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            mlflow.sklearn.log_model(model, "model")
            mlflow.log_metric("accuracy", accuracy_score(y_train, model.predict(X_train)))
            return model
    except Exception as e:
        raise Exception(f"Failed to train model: {e}")
