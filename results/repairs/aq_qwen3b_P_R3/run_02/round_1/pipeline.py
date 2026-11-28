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
        raise ValueError("The file does not exist.")
    except pd.errors.EmptyDataError:
        raise ValueError("The file is empty.")

def preprocess_data(data):
    """Preprocess the data."""
    # Drop columns with missing values
    data = data.dropna()
    
    # Convert categorical columns to numerical
    data['Hour'] = data['Hour'].astype('category').cat.codes
    data['DayOfWeek'] = data['DayOfWeek'].astype('category').cat.codes
    data['Month'] = data['Month'].astype('category').cat.codes
    
    # Separate features and target
    X = data.drop('Anomaly', axis=1)
    y = data['Anomaly']
    
    return X, y

def train_model(X, y):
    """Train a RandomForest classifier."""
    try:
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        # Log the model to MLflow
        mlflow.start_run()
        mlflow.log_param("n_estimators", model.n_estimators)
        mlflow.sklearn.log_model(model, "random_forest_model")
        
        # Evaluate the model
        y_pred = model.predict(X)
        accuracy = accuracy_score(y, y_pred)
        mlflow.log_metric("accuracy", accuracy)
        
        mlflow.end_run()
        
        return model
    except Exception as e:
        mlflow.log_exception(e)
        raise e
