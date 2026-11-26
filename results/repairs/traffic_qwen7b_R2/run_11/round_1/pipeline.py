import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import mlflow
import mlflow.sklearn

def load_data(file_path):
    """Load data from a CSV file."""
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        raise Exception(f"Failed to load data: {e}")

def preprocess_data(data):
    """Preprocess data to prepare for modeling."""
    try:
        # Drop any rows with missing values
        data = data.dropna()
        
        # Convert categorical variables to dummy variables
        data = pd.get_dummies(data, columns=['weather_main', 'IsHoliday'], drop_first=True)
        
        # Separate features and target
        X = data.drop(columns=['Congestion'])
        y = data['Congestion']
        
        return X, y
    except Exception as e:
        raise Exception(f"Failed to preprocess data: {e}")

def train_model(X_train, y_train):
    """Train a RandomForest model."""
    try:
        with mlflow.start_run():
            # Set MLflow experiment name
            mlflow.set_experiment("traffic_congestion_detection")
            
            # Initialize the RandomForestClassifier
            rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
            
            # Train the model
            rf_model.fit(X_train, y_train)
            
            # Log the model
            mlflow.sklearn.log_model(rf_model, "model")
            
            # Log model parameters
            mlflow.log_param("n_estimators", 100)
            mlflow.log_param("random_state", 42)
            
            # Log model metrics
            y_pred = rf_model.predict(X_train)
            accuracy = accuracy_score(y_train, y_pred)
            classification_rep = classification_report(y_train, y_pred)
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_text(classification_rep, "classification_report.txt")
            
            return rf_model
    except Exception as e:
        raise Exception(f"Failed to train model: {e}")
