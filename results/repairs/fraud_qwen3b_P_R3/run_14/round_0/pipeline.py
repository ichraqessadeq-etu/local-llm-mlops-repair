import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import mlflow
import mlflow.sklearn

def load_data(file_path):
    """Load the dataset from a CSV file."""
    try:
        data = pd.read_csv(file_path)
        return data
    except FileNotFoundError:
        raise ValueError("File not found")

def preprocess_data(data):
    """Preprocess the data."""
    try:
        # Assuming 'Class' is the target column
        X = data.drop('Class', axis=1)
        y = data['Class']
        
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        return X_train, X_test, y_train, y_test
    except KeyError:
        raise ValueError("Target column 'Class' not found")

def train_model(X_train, y_train):
    """Train the RandomForest model."""
    try:
        with mlflow.start_run():
            # Initialize the model
            model = RandomForestClassifier(random_state=42)
            
            # Train the model
            model.fit(X_train, y_train)
            
            # Log the model
            mlflow.sklearn.log_model(model, "random_forest_model")
            
            # Log metrics
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            classification_report_str = classification_report(y_test, y_pred)
            
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_text("classification_report", classification_report_str)
            
            return model
    except Exception as e:
        mlflow.log_exception(e)
        raise e
