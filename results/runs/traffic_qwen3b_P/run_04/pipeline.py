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
        raise FileNotFoundError(f"File {file_path} not found.")
    except pd.errors.EmptyDataError:
        raise pd.errors.EmptyDataError("The file is empty.")

def preprocess_data(data):
    """Preprocess the data."""
    try:
        # Convert 'IsHoliday' to binary
        data['IsHoliday'] = data['IsHoliday'].astype(int)
        
        # Define features and target
        X = data[['temp', 'rain_1h', 'snow_1h', 'clouds_all', 'weather_main', 'IsHoliday', 'Hour', 'DayOfWeek', 'Month']]
        y = data['Congestion']
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        return X_train, X_test, y_train, y_test
    except KeyError as e:
        raise KeyError(f"Missing column(s): {e}")

def train_model(X_train, y_train):
    """Train a RandomForest model."""
    try:
        with mlflow.start_run():
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            # Log model
            mlflow.sklearn.log_model(model, "congestion_detector")
            
            # Log metrics
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            mlflow.log_metric("accuracy", accuracy)
            
            return model
    except Exception as e:
        raise Exception(f"An error occurred during training: {e}")
