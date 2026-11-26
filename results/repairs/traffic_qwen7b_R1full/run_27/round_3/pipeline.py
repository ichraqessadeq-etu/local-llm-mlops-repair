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
    """Preprocess data to separate features and target."""
    # Convert non-numeric columns to numeric, filling missing values with 0
    data['rain_1h'] = pd.to_numeric(data['rain_1h'], errors='coerce').fillna(0)
    data['snow_1h'] = pd.to_numeric(data['snow_1h'], errors='coerce').fillna(0)
    data['clouds_all'] = pd.to_numeric(data['clouds_all'], errors='coerce').fillna(0)
    
    X = data.drop(columns=['Congestion'])
    y = data['Congestion']
    return X, y

def train_model(X_train, y_train):
    """Train a RandomForestClassifier model."""
    with mlflow.start_run():
        mlflow.log_param("model", "RandomForestClassifier")
        model = RandomForestClassifier(random_state=42)  # Add random_state for reproducibility
        model.fit(X_train, y_train)
        y_pred = model.predict(X_train)
        accuracy = accuracy_score(y_train, y_pred)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.sklearn.log_model(model, "model")
    return model
