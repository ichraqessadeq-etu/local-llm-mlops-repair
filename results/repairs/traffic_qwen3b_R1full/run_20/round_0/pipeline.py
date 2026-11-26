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
        print(f"Error: File {file_path} not found.")
        return None

def preprocess_data(data):
    """Preprocess the data."""
    try:
        # Convert categorical variables to numerical
        data['weather_main'] = data['weather_main'].map({'Clear': 0, 'Clouds': 1, 'Rain': 2, 'Snow': 3})
        data['IsHoliday'] = data['IsHoliday'].astype(int)
        
        # Separate features and target
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
        with mlflow.start_run():
            mlflow.sklearn.log_model(model, "congestion_detector")
            mlflow.log_param("n_estimators", model.n_estimators)
            mlflow.log_metric("accuracy", accuracy_score(y_train, model.predict(X_train)))
        
        return model
    except Exception as e:
        print(f"Error training model: {e}")
        return None
