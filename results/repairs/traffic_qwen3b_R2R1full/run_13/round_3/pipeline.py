import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
from mlflow.exceptions import MlflowException

def load_data(file_path):
    """Load data from a CSV file."""
    try:
        data = pd.read_csv(file_path)
        return data
    except FileNotFoundError:
        raise MlflowException("File not found: {}".format(file_path))

def preprocess_data(data):
    """Preprocess the data by splitting into features and target, and splitting into training and testing sets."""
    features = ['temp', 'rain_1h', 'snow_1h', 'clouds_all', 'weather_main', 'IsHoliday', 'Hour', 'DayOfWeek', 'Month']
    target = 'Congestion'
    
    X = data[features]
    y = data[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    """Train a RandomForest model and log it using MLflow."""
    try:
        with mlflow.start_run():
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            # Log the model
            mlflow.sklearn.log_model(model, 'congestion_detector')
            
            # Log metrics
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            mlflow.log_metric('accuracy', accuracy)
            
            return model
    except Exception as e:
        mlflow.log_exception(e)
        raise
