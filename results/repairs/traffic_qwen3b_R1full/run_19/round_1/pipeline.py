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
        raise pd.errors.EmptyDataError("The file is empty or has no data.")

def preprocess_data(data):
    """Preprocess the data."""
    try:
        # Convert 'IsHoliday' to binary
        data['IsHoliday'] = data['IsHoliday'].astype(int)
        
        # Encode categorical variables
        data = pd.get_dummies(data, columns=['weather_main'])
        
        # Split features and target
        X = data.drop('Congestion', axis=1)
        y = data['Congestion']
        
        return X, y
    except Exception as e:
        raise Exception(f"Preprocessing error: {e}")

def train_model(X_train, y_train):
    """Train the RandomForest model."""
    try:
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Log the model to MLflow
        with mlflow.start_run(run_name="RandomForest_Congestion_Detector"):
            mlflow.log_param("n_estimators", 100)
            mlflow.log_param("random_state", 42)
            mlflow.sklearn.log_model(model, "random_forest_model")
        
        return model
    except Exception as e:
        raise Exception(f"Model training error: {e}")
