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
        raise Exception(f"Failed to load data: {e}")

def preprocess_data(data):
    """Preprocess data by separating features and target."""
    try:
        # Drop rows with missing values in the target column
        data = data.dropna(subset=['Congestion'])
        
        # Convert 'Rain_1h' to numeric, handling non-numeric values
        data['Rain_1h'] = pd.to_numeric(data['Rain_1h'], errors='coerce')
        
        # Drop rows with missing values in 'Rain_1h'
        data = data.dropna(subset=['Rain_1h'])
        
        X = data.drop(columns=['Congestion'])
        y = data['Congestion']
        return X, y
    except Exception as e:
        raise Exception(f"Failed to preprocess data: {e}")

def train_model(X_train, y_train):
    """Train a RandomForestClassifier."""
    try:
        with mlflow.start_run():
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            mlflow.log_param("n_estimators", 100)
            mlflow.sklearn.log_model(model, "model")
            return model
    except Exception as e:
        raise Exception(f"Failed to train model: {e}")
