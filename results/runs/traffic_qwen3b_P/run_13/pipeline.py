import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from mlflow import log_metric, log_param, log_artifact
import mlflow

def load_data(file_path):
    """Load data from a CSV file."""
    try:
        data = pd.read_csv(file_path)
        return data
    except FileNotFoundError:
        log_metric("error", "File not found")
        raise
    except pd.errors.EmptyDataError:
        log_metric("error", "Empty data")
        raise

def preprocess_data(data):
    """Preprocess the data."""
    try:
        # Convert categorical variables to numerical
        data['weather_main'] = data['weather_main'].map({'Clear': 0, 'Clouds': 1, 'Rain': 2, 'Snow': 3})
        data['IsHoliday'] = data['IsHoliday'].map({True: 1, False: 0})

        # Split features and target
        X = data.drop('Congestion', axis=1)
        y = data['Congestion']

        # Split into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        return X_train, X_test, y_train, y_test
    except Exception as e:
        log_metric("error", str(e))
        raise

def train_model(X_train, y_train):
    """Train a RandomForest model."""
    try:
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Log model parameters
        log_param("n_estimators", model.n_estimators)
        log_param("random_state", model.random_state)

        # Evaluate the model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        log_metric("accuracy", accuracy)

        # Log the model
        with mlflow.start_run():
            mlflow.sklearn.log_model(model, "rf_model")

        return model
    except Exception as e:
        log_metric("error", str(e))
        raise
