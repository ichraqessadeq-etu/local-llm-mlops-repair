# pipeline.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import mlflow
import mlflow.sklearn
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

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
        data.drop(['PT08.S1(CO)', 'PT08.S2(NMHC)', 'PT08.S3(NOx)', 'PT08.S4(NO2)', 'PT08.S5(O3)', 'NOx(GT)', 'C6H6(GT)', 'AH'], axis=1, inplace=True)
        
        # Convert categorical features to numerical
        data['Hour'] = pd.Categorical(data['Hour']).codes
        data['DayOfWeek'] = pd.Categorical(data['DayOfWeek']).codes
        data['Month'] = pd.Categorical(data['Month']).codes
        
        # Split data into features (X) and target (y)
        X = data[['PT08.S1(CO)', 'PT08.S2(NMHC)', 'PT08.S3(NOx)', 'PT08.S4(NO2)', 'PT08.S5(O3)', 'NOx(GT)', 'C6H6(GT)', 'T', 'RH', 'Hour', 'DayOfWeek', 'Month']]
        y = data['Anomaly']
        
        # Scale features
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        return X, y
    except Exception as e:
        logging.error(f"Error preprocessing data: {e}")
        raise

def train_model(X_train, y_train):
    try:
        # Train RandomForest model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Log model to MLflow
        mlflow.sklearn.log_model(model, "random_forest_model")
        
        return model
    except Exception as e:
        logging.error(f"Error training model: {e}")
        raise

def main():
    data = load_data('data/air_quality.csv')
    X, y = preprocess_data(data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = train_model(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = model.score(X_test, y_test)
    logging.info(f"Model accuracy: {accuracy}")
    
    return model

if __name__ == "__main__":
    model = main()
