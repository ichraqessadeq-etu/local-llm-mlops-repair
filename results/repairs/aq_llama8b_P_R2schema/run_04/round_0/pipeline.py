import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def load_data(file_path):
    """Loads the air quality data from the CSV file."""
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        mlflow.log_error(f"Error loading data: {e}")
        raise

def preprocess_data(data):
    """Preprocesses the data by encoding categorical variables and scaling numeric variables."""
    try:
        X = data.drop('Anomaly', axis=1)
        y = data['Anomaly']
        X = pd.get_dummies(X, columns=['Hour', 'DayOfWeek', 'Month'])
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        return X, y
    except Exception as e:
        mlflow.log_error(f"Error preprocessing data: {e}")
        raise

def train_model(X_train, y_train):
    """Trains a RandomForest classifier on the preprocessed data."""
    try:
        model = RandomForestClassifier(n_estimators=100)
        model.fit(X_train, y_train)
        mlflow.sklearn.log_model(model, "random_forest")
        return model
    except Exception as e:
        mlflow.log_error(f"Error training model: {e}")
        raise
