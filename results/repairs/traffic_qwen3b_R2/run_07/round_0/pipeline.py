import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
import warnings
import os

warnings.filterwarnings("ignore")

def load_data(file_path):
    """Load data from the specified file path."""
    try:
        data = pd.read_csv(file_path)
        return data
    except FileNotFoundError:
        mlflow.log_exception("Data file not found.")
        raise
    except pd.errors.EmptyDataError:
        mlflow.log_exception("Data file is empty.")
        raise

def preprocess_data(data):
    """Preprocess the data."""
    try:
        # Convert target column to binary
        data['Congestion'] = data['Congestion'].map({'high': 1, 'low': 0})
        
        # Drop irrelevant columns
        data = data.drop(['weather_main', 'IsHoliday'], axis=1)
        
        # Split features and target
        X = data.drop('Congestion', axis=1)
        y = data['Congestion']
        
        # Split into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        return X_train, y_train
    except Exception as e:
        mlflow.log_exception(f"Preprocessing error: {e}")
        raise

def train_model(X_train, y_train):
    """Train the RandomForest model."""
    try:
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Log model to MLflow
        mlflow.sklearn.log_model(model, "random_forest_model")
        
        return model
    except Exception as e:
        mlflow.log_exception(f"Model training error: {e}")
        raise

def main():
    file_path = 'data/traffic.csv'
    data = load_data(file_path)
    X_train, y_train = preprocess_data(data)
    model = train_model(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    mlflow.log_metric("accuracy", accuracy)
    
    print(f"Model accuracy: {accuracy}")

if __name__ == "__main__":
    main()
