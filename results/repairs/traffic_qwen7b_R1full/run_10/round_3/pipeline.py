import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient

def load_data(file_path):
    """Load data from a CSV file."""
    return pd.read_csv(file_path)

def preprocess_data(data):
    """Preprocess the data."""
    # Drop any rows with missing values
    data = data.dropna()
    
    # Convert categorical variables to numerical
    data['weather_main'] = data['weather_main'].astype('category').cat.codes
    
    # Separate features and target
    X = data.drop(columns=['Congestion'])
    y = data['Congestion']
    
    return X, y

def train_model(X_train, y_train):
    """Train a RandomForestClassifier."""
    with mlflow.start_run():
        # Set experiment name
        mlflow.set_experiment("RandomForestCongestionDetector")
        
        # Log parameters
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("max_depth", 10)
        
        # Train the model
        model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        model.fit(X_train, y_train)
        
        # Log model
        mlflow.sklearn.log_model(model, "model")
        
        # Log metrics
        y_pred = model.predict(X_train)
        accuracy = accuracy_score(y_train, y_pred)
        mlflow.log_metric("accuracy", accuracy)
        
        # Infer signature
        signature = infer_signature(X_train, y_pred)
        
        # Save model with signature
        mlflow.sklearn.save_model(model, "model", signature=signature)
        
        return model

def load_model():
    """Load the trained model."""
    client = MlflowClient()
    model_uri = "models:/RandomForestCongestionDetector/1"
    model = mlflow.sklearn.load_model(model_uri)
    return model
