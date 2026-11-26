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
        raise ValueError(f"File not found: {file_path}")

def preprocess_data(data):
    """Preprocess the data."""
    # Convert categorical variables to numerical
    data['weather_main'] = data['weather_main'].map({'Clear': 0, 'Clouds': 1, 'Rain': 2, 'Snow': 3})
    data['IsHoliday'] = data['IsHoliday'].map({False: 0, True: 1})
    
    # Define features and target
    X = data.drop('Congestion', axis=1)
    y = data['Congestion']
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    """Train the RandomForest model."""
    try:
        with mlflow.start_run():
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            # Log the model
            mlflow.sklearn.log_model(model, "congestion_detector")
            
            # Evaluate the model
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            mlflow.log_metric("accuracy", accuracy)
            
            return model
    except Exception as e:
        mlflow.log_exception(e)
        raise e
