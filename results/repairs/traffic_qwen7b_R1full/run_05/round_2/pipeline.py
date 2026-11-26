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
    """Preprocess data and split into features and target."""
    # Convert non-numeric columns to numeric
    data['weather_main'] = data['weather_main'].astype('category').cat.codes
    data['IsHoliday'] = data['IsHoliday'].astype('category').cat.codes
    
    X = data.drop(columns=['Congestion'])
    y = data['Congestion']
    return X, y

def train_model(X_train, y_train):
    """Train a RandomForestClassifier and log it using MLflow."""
    with mlflow.start_run():
        mlflow.log_param("model", "RandomForestClassifier")
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        mlflow.sklearn.log_model(model, "model")
        mlflow.log_metric("accuracy", accuracy_score(y_train, model.predict(X_train)))
        return model

def evaluate_model(model, X_test, y_test):
    """Evaluate the model on test data and log the accuracy."""
    with mlflow.start_run():
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        mlflow.log_metric("test_accuracy", accuracy)
        return accuracy
