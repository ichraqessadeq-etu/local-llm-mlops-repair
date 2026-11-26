import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
from mlflow import log_param, log_metric, log_artifact

def load_data(file_path):
    """Load data from a CSV file."""
    return pd.read_csv(file_path)

def preprocess_data(data):
    """Preprocess the data to prepare it for modeling."""
    # Handle missing values
    data.fillna(method='ffill', inplace=True)
    
    # Convert categorical variables to numerical
    data['weather_main'] = data['weather_main'].astype('category').cat.codes
    
    # Split data into features and target
    X = data.drop(columns=['Congestion'])
    y = data['Congestion']
    
    return X, y

def train_model(X_train, y_train):
    """Train a RandomForestClassifier model."""
    with mlflow.start_run():
        # Log parameters
        log_param("n_estimators", 100)
        log_param("max_depth", 10)
        
        # Initialize and train the model
        model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        model.fit(X_train, y_train)
        
        # Log model
        mlflow.sklearn.log_model(model, "model")
        
        # Log metrics
        y_pred = model.predict(X_train)
        accuracy = accuracy_score(y_train, y_pred)
        log_metric("accuracy", accuracy)
        
        # Log artifacts
        mlflow.log_artifact("data/traffic.csv", artifact_path="data")
        
        return model
