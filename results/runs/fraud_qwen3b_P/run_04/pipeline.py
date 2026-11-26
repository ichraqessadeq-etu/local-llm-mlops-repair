import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
import mlflow
import mlflow.sklearn

def load_data(file_path):
    """Load data from a CSV file."""
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        raise ValueError(f"File {file_path} not found.")
    except pd.errors.ParserError:
        raise ValueError(f"Error parsing file {file_path}.")

def preprocess_data(data):
    """Preprocess the data."""
    try:
        # Separate features and target
        X = data.drop('Class', axis=1)
        y = data['Class']
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Standardize the features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        return X_train, X_test, y_train, y_test
    except Exception as e:
        raise ValueError(f"Error preprocessing data: {e}")

def train_model(X_train, y_train):
    """Train a RandomForest model."""
    try:
        # Initialize the model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        
        # Train the model
        model.fit(X_train, y_train)
        
        return model
    except Exception as e:
        raise ValueError(f"Error training model: {e}")

def log_model(model, experiment_name, artifact_path):
    """Log the model using MLflow."""
    try:
        # Start a new MLflow run
        with mlflow.start_run(experiment_name=experiment_name):
            # Log the model
            mlflow.sklearn.log_model(model, artifact_path=artifact_path)
            
            # Log metrics
            y_pred = model.predict(X_test)
            report = classification_report(y_test, y_pred)
            mlflow.log_text(report, "classification_report.txt")
    except Exception as e:
        raise ValueError(f"Error logging model: {e}")
