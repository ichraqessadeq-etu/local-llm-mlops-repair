import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from mlflow import logging
from mlflow.tracking import MlflowClient

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
        data.drop(['Hour', 'DayOfWeek', 'Month'], axis=1, inplace=True)
        
        # Scale features using StandardScaler
        scaler = StandardScaler()
        data[['PT08.S1(CO)', 'PT08.S2(NMHC)', 'PT08.S3(NOx)', 'PT08.S4(NO2)', 'PT08.S5(O3)', 'NOx(GT)', 'C6H6(GT)', 'T', 'RH', 'AH']] = scaler.fit_transform(data[['PT08.S1(CO)', 'PT08.S2(NMHC)', 'PT08.S3(NOx)', 'PT08.S4(NO2)', 'PT08.S5(O3)', 'NOx(GT)', 'C6H6(GT)', 'T', 'RH', 'AH']])
        
        # Split data into features (X) and target (y)
        X = data.drop('Anomaly', axis=1)
        y = data['Anomaly']
        
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        return X_train, X_test, y_train, y_test
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

def evaluate_model(model, X_test, y_test):
    try:
        # Make predictions on test set
        y_pred = model.predict(X_test)
        
        # Evaluate model performance
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        return y_pred
    except Exception as e:
        logging.error(f"Error evaluating model: {e}")
        raise
