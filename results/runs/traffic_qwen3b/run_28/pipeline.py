import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def load_data(file_path):
    """Load data from a CSV file."""
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        raise

def preprocess_data(data):
    """Preprocess the data."""
    try:
        # Convert 'IsHoliday' to binary
        data['IsHoliday'] = data['IsHoliday'].map({True: 1, False: 0})
        
        # Encode categorical variables
        data = pd.get_dummies(data, columns=['weather_main', 'Hour', 'DayOfWeek', 'Month'])
        
        # Split features and target
        X = data.drop('Congestion', axis=1)
        y = data['Congestion']
        
        return X, y
    except Exception as e:
        logger.error(f"Failed to preprocess data: {e}")
        raise

def train_model(X_train, y_train):
    """Train the RandomForest model."""
    try:
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Log the model to MLflow
        mlflow.log_param("n_estimators", model.n_estimators)
        mlflow.log_param("random_state", model.random_state)
        
        # Log the model
        mlflow.sklearn.log_model(model, "random_forest_model")
        
        return model
    except Exception as e:
        logger.error(f"Failed to train model: {e}")
        raise
