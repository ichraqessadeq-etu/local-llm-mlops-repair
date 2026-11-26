import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
import logging

logging.basicConfig(level=logging.INFO)

def load_data(file_path):
    """Load data from a CSV file."""
    logging.info("Loading data from %s", file_path)
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        logging.error("Error loading data: %s", e)
        raise

def preprocess_data(data):
    """Preprocess the data."""
    logging.info("Preprocessing data")
    try:
        # Drop columns with missing values
        data.dropna(inplace=True)
        
        # Convert categorical variables to numerical
        data['weather_main'] = data['weather_main'].astype('category').cat.codes
        
        # Convert DayOfWeek to numeric
        data['DayOfWeek'] = data['DayOfWeek'].map({'Monday': 1, 'Tuesday': 2, 'Wednesday': 3, 'Thursday': 4, 'Friday': 5, 'Saturday': 6, 'Sunday': 7})
        
        # Split features and target
        X = data.drop('Congestion', axis=1)
        y = data['Congestion']
        
        return X, y
    except Exception as e:
        logging.error("Error preprocessing data: %s", e)
        raise

def train_model(X_train, y_train):
    """Train a RandomForest model."""
    logging.info("Training model")
    try:
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        return model
    except Exception as e:
        logging.error("Error training model: %s", e)
        raise

def evaluate_model(model, X_test, y_test):
    """Evaluate the model."""
    logging.info("Evaluating model")
    try:
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        logging.info("Accuracy: %.2f", accuracy)
        return accuracy
    except Exception as e:
        logging.error("Error evaluating model: %s", e)
        raise

def run_pipeline(file_path):
    """Run the entire pipeline."""
    try:
        data = load_data(file_path)
        X, y = preprocess_data(data)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        with mlflow.start_run():
            model = train_model(X_train, y_train)
            evaluate_model(model, X_test, y_test)
        
        logging.info("Pipeline completed successfully")
    except Exception as e:
        logging.error("Pipeline failed: %s", e)
        raise
