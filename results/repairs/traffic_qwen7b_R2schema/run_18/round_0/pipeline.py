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
    except Exception as e:
        raise ValueError(f"Failed to load data: {e}")

def preprocess_data(data):
    """Preprocess the data."""
    try:
        # Handle categorical variables
        data['weather_main'] = data['weather_main'].astype('category').cat.codes
        
        # Split data into features and target
        X = data.drop(columns=['Congestion'])
        y = data['Congestion']
        
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        return X_train, X_test, y_train, y_test
    except Exception as e:
        raise ValueError(f"Failed to preprocess data: {e}")

def train_model(X_train, y_train):
    """Train the RandomForest model."""
    try:
        # Initialize MLflow experiment
        mlflow.set_experiment("traffic-congestion-detection")
        
        # Start MLflow run
        with mlflow.start_run():
            # Set parameters
            mlflow.log_param("n_estimators", 100)
            mlflow.log_param("max_depth", 10)
            
            # Train model
            model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
            model.fit(X_train, y_train)
            
            # Log model
            mlflow.sklearn.log_model(model, "model")
            
            # Log metrics
            y_pred = model.predict(X_train)
            accuracy = accuracy_score(y_train, y_pred)
            mlflow.log_metric("train_accuracy", accuracy)
            
            return model
    except Exception as e:
        raise ValueError(f"Failed to train model: {e}")
