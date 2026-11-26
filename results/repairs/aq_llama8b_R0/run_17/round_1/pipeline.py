# pipeline.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import mlflow
import logging
from sklearn.metrics import accuracy_score, classification_report

def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise

def preprocess_data(data):
    try:
        # Drop unnecessary columns
        data.drop(columns=['Hour', 'DayOfWeek', 'Month'], inplace=True)
        
        # Scale features using StandardScaler
        scaler = StandardScaler()
        X = scaler.fit_transform(data.drop('Anomaly', axis=1))
        
        # Encode target variable
        y = data['Anomaly'].map({'Normal': 0, 'Anomaly': 1})
        
        return X, y
    except Exception as e:
        logging.error(f"Error preprocessing data: {e}")
        raise

def train_model(X, y):
    try:
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train RandomForest model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Make predictions on test set
        y_pred = model.predict(X_test)
        
        # Evaluate model performance
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        
        # Log model with MLflow
        mlflow.sklearn.log_model(model, "random_forest_model")
        
        # Log metrics with MLflow
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("classification_report", report)
        
        return model, accuracy, report
    except Exception as e:
        logging.error(f"Error training model: {e}")
        raise

def main():
    data = load_data('data/air_quality.csv')
    X, y = preprocess_data(data)
    model, accuracy, report = train_model(X, y)
    
    print(f"Model accuracy: {accuracy}")
    print(f"Classification report:\n{report}")

if __name__ == "__main__":
    main()
