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
    except FileNotFoundError:
        raise ValueError("File not found.")
    except pd.errors.ParserError:
        raise ValueError("Error parsing CSV file.")

def preprocess_data(data):
    """Preprocess the data by encoding categorical variables and splitting into features and target."""
    try:
        # Encode categorical variables
        data['weather_main'] = pd.Categorical(data['weather_main'])
        data['IsHoliday'] = pd.Categorical(data['IsHoliday'])
        
        # Define features and target
        X = data.drop('Congestion', axis=1)
        y = data['Congestion']
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        return X_train, X_test, y_train, y_test
    except Exception as e:
        raise Exception(f"Error preprocessing data: {e}")

def train_model(X_train, y_train):
    """Train a RandomForestClassifier model."""
    try:
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        return model
    except Exception as e:
        raise Exception(f"Error training model: {e}")

def save_model(model, experiment_name, run_name):
    """Save the model using MLflow."""
    try:
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run(run_name=run_name):
            mlflow.sklearn.log_model(model, "model")
            print("Model saved successfully.")
    except Exception as e:
        raise Exception(f"Error saving model: {e}")
