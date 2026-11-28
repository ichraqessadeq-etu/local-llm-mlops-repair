import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

def load_data(file_path):
    """Load data from a CSV file."""
    try:
        logging.info("Loading data from file: %s", file_path)
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        logging.error("Error loading data: %s", str(e))
        raise

def preprocess_data(data):
    """Preprocess the data."""
    try:
        logging.info("Preprocessing data")
        # Handle missing values
        data.fillna(method='ffill', inplace=True)
        
        # Convert categorical variables to numerical
        data['weather_main'] = data['weather_main'].astype('category').cat.codes
        
        # Define features and target
        X = data.drop('Congestion', axis=1)
        y = data['Congestion']
        
        return X, y
    except Exception as e:
        logging.error("Error preprocessing data: %s", str(e))
        raise

def train_model(X_train, y_train):
    """Train the model."""
    try:
        logging.info("Training model")
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        return model
    except Exception as e:
        logging.error("Error training model: %s", str(e))
        raise

def main():
    file_path = 'data/traffic.csv'
    try:
        data = load_data(file_path)
        X, y = preprocess_data(data)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        with mlflow.start_run():
            model = train_model(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            logging.info("Model accuracy: %.2f", accuracy)
            
            mlflow.log_param("n_estimators", 100)
            mlflow.log_metric("accuracy", accuracy)
            mlflow.sklearn.log_model(model, "rf_model")
    except Exception as e:
        logging.error("Error in pipeline: %s", str(e))

if __name__ == "__main__":
    main()
