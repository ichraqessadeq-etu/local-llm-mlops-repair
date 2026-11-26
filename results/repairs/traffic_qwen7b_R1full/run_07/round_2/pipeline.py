import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
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
    return pd.read_csv(file_path)

def preprocess_data(data):
    """
    Preprocess the data by separating features and target.
    
    Args:
        data (pd.DataFrame): Input data.
    
    Returns:
        tuple: Tuple containing features (X) and target (y).
    """
    features = ['temp', 'rain_1h', 'snow_1h', 'clouds_all', 'weather_main', 'IsHoliday', 'Hour', 'DayOfWeek', 'Month']
    target = 'Congestion'
    X = data[features]
    y = data[target]
    return X, y

def train_model(X_train, y_train):
    """
    Train a RandomForestClassifier model.
    
    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.
    
    Returns:
        RandomForestClassifier: Trained model.
    """
    with mlflow.start_run():
        mlflow.log_param("model", "RandomForestClassifier")
        model = RandomForestClassifier(random_state=42)  # Added random_state for reproducibility
        model.fit(X_train, y_train)
        mlflow.sklearn.log_model(model, "model")
        mlflow.log_metric("accuracy", accuracy_score(y_train, model.predict(X_train)))
        mlflow.log_artifact("data/traffic.csv", artifact_path="data")
        return model
