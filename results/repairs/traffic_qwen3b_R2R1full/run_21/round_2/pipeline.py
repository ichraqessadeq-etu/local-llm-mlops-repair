import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import mlflow
import mlflow.sklearn

def load_data(file_path):
    """Load data from a CSV file."""
    try:
        data = pd.read_csv(file_path)
        return data
    except FileNotFoundError:
        raise ValueError(f"File not found: {file_path}")
    except pd.errors.EmptyDataError:
        raise ValueError("The file is empty or contains no data.")

def preprocess_data(data):
    """Preprocess the data."""
    try:
        # Convert 'IsHoliday' to binary
        data['IsHoliday'] = data['IsHoliday'].map({True: 1, False: 0})
        
        # Encode categorical variables
        data = pd.get_dummies(data, columns=['weather_main', 'DayOfWeek', 'Month'])
        
        # Split features and target
        X = data.drop('Congestion', axis=1)
        y = data['Congestion']
        
        return X, y
    except Exception as e:
        raise ValueError(f"Preprocessing error: {e}")

def train_model(X_train, y_train):
    """Train the model."""
    try:
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Log the model to MLflow
        mlflow.sklearn.log_model(model, "random_forest_model")
        
        return model
    except Exception as e:
        raise ValueError(f"Training error: {e}")

