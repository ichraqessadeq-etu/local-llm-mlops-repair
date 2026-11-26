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
        print(f"File not found: {file_path}")
        return None

def preprocess_data(data):
    """Preprocess the data."""
    try:
        # Handle missing values
        data = data.dropna()
        
        # Convert categorical variables to numerical
        data = pd.get_dummies(data, columns=['weather_main', 'IsHoliday'])
        
        # Define features and target
        X = data.drop('Congestion', axis=1)
        y = data['Congestion']
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        return X_train, X_test, y_train, y_test
    except Exception as e:
        print(f"Error preprocessing data: {e}")
        return None, None, None, None

def train_model(X_train, y_train):
    """Train a RandomForest model."""
    try:
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Log the model to MLflow
        mlflow.log_param("n_estimators", 100)
        mlflow.sklearn.log_model(model, "random_forest_model")
        
        return model
    except Exception as e:
        print(f"Error training model: {e}")
        return None

def evaluate_model(model, X_test, y_test):
    """Evaluate the model."""
    try:
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model accuracy: {accuracy:.2f}")
        return accuracy
    except Exception as e:
        print(f"Error evaluating model: {e}")
        return None
