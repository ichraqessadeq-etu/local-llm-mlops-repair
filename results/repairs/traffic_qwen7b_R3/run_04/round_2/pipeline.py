import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn

def load_data(file_path):
    """Load data from a CSV file."""
    return pd.read_csv(file_path)

def preprocess_data(data):
    """Preprocess data for model training."""
    # Drop rows with missing values
    data = data.dropna()
    
    # Encode categorical variables
    data = pd.get_dummies(data, columns=['weather_main'], drop_first=True)
    
    # Separate features and target
    X = data.drop(columns=['Congestion'])
    y = data['Congestion']
    
    return X, y

def train_model(X_train, y_train):
    """Train a RandomForestClassifier."""
    with mlflow.start_run():
        # Set MLflow experiment name
        mlflow.set_experiment("RandomForestCongestionDetector")
        
        # Initialize and train the model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Log model and metrics
        mlflow.sklearn.log_model(model, "model")
        mlflow.log_param("n_estimators", 100)
        mlflow.log_metric("accuracy", accuracy_score(y_train, model.predict(X_train)))
        
        return model
