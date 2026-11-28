import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
import logging

logging.basicConfig(level=logging.INFO)

def load_data(file_path):
    try:
        logging.info("Loading data from file: %s", file_path)
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        logging.error("Error loading data: %s", e)
        raise

def preprocess_data(data):
    try:
        logging.info("Preprocessing data")
        # Drop rows with missing values
        data = data.dropna()
        
        # Convert categorical variables to numerical
        data['DayOfWeek'] = pd.Categorical(data['DayOfWeek']).codes
        data['Month'] = pd.Categorical(data['Month']).codes
        
        # Define features and target
        X = data.drop('Anomaly', axis=1)
        y = data['Anomaly']
        
        return X, y
    except Exception as e:
        logging.error("Error preprocessing data: %s", e)
        raise

def train_model(X_train, y_train):
    try:
        logging.info("Training model")
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        return model
    except Exception as e:
        logging.error("Error training model: %s", e)
        raise

def evaluate_model(model, X_test, y_test):
    try:
        logging.info("Evaluating model")
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        logging.info("Model accuracy: %.2f", accuracy)
        return accuracy
    except Exception as e:
        logging.error("Error evaluating model: %s", e)
        raise

def main():
    try:
        # Load data
        data = load_data('data/air_quality.csv')
        
        # Preprocess data
        X, y = preprocess_data(data)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        model = train_model(X_train, y_train)
        
        # Evaluate model
        evaluate_model(model, X_test, y_test)
        
        # Log model to MLflow
        mlflow.set_experiment("air_quality_anomaly_detection")
        mlflow.sklearn.log_model(model, "model")
    except Exception as e:
        logging.error("Error in pipeline: %s", e)

if __name__ == "__main__":
    main()
