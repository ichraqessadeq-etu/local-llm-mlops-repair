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
        raise FileNotFoundError(f"File not found: {file_path}")
    except pd.errors.EmptyDataError:
        raise pd.errors.EmptyDataError("The file is empty")

def preprocess_data(data):
    """Preprocess the data."""
    try:
        # Handle missing values
        data.fillna(data.mean(), inplace=True)
        
        # Convert categorical variables to numerical
        data['weather_main'] = data['weather_main'].map({
            'Clear': 0,
            'Clouds': 1,
            'Rain': 2,
            'Snow': 3,
            'Thunderstorm': 4
        })
        
        # Define features and target
        X = data.drop('Congestion', axis=1)
        y = data['Congestion']
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        return X_train, X_test, y_train, y_test
    except Exception as e:
        raise Exception(f"Preprocessing failed: {e}")

def train_model(X_train, y_train):
    """Train a RandomForest model."""
    try:
        with mlflow.start_run():
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(X_train, y_train)
            
            # Log model
            mlflow.sklearn.log_model(rf, 'random_forest_model')
            
            # Evaluate model
            y_pred = rf.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            mlflow.log_metric('accuracy', accuracy)
            
            return rf
    except Exception as e:
        raise Exception(f"Model training failed: {e}")
