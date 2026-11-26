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
    """Preprocess data to split into features and target."""
    # Convert categorical columns to numerical using one-hot encoding
    data = pd.get_dummies(data, columns=['weather_main', 'IsHoliday'], drop_first=True)
    
    X = data.drop(columns=['Congestion'])
    y = data['Congestion']
    return X, y

def train_model(X_train, y_train):
    """Train a RandomForestClassifier and log the model using MLflow."""
    with mlflow.start_run():
        mlflow.log_param("model", "RandomForestClassifier")
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        mlflow.sklearn.log_model(model, "model")
        y_pred = model.predict(X_train)
        accuracy = accuracy_score(y_train, y_pred)
        mlflow.log_metric("train_accuracy", accuracy)
        return model
