import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow

def load_data(file_path):
    """Load data from a CSV file."""
    try:
        data = pd.read_csv(file_path)
        return data
    except FileNotFoundError:
        raise ValueError(f"File not found: {file_path}")

def preprocess_data(data):
    """Preprocess the data."""
    try:
        # Convert 'IsHoliday' to binary
        data['IsHoliday'] = data['IsHoliday'].astype(int)
        
        # One-hot encode categorical features
        data = pd.get_dummies(data, columns=['weather_main', 'DayOfWeek', 'Month'])
        
        # Define features and target
        X = data.drop('Congestion', axis=1)
        y = data['Congestion']
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        return X_train, X_test, y_train, y_test
    except Exception as e:
        raise ValueError(f"Preprocessing error: {e}")

def train_model(X_train, y_train):
    """Train the RandomForest model."""
    try:
        with mlflow.start_run():
            # Initialize the model
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            
            # Train the model
            model.fit(X_train, y_train)
            
            # Log model parameters
            mlflow.log_param("n_estimators", model.n_estimators)
            mlflow.log_param("random_state", model.random_state)
            
            # Evaluate the model
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            mlflow.log_metric("accuracy", accuracy)
            
            return model
    except Exception as e:
        raise ValueError(f"Training error: {e}")
