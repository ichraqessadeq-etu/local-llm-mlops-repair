import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from mlflow import log_param, log_metric
from mlflow.sklearn import log_model

def load_data(file_path):
    """Load data from a CSV file."""
    return pd.read_csv(file_path)

def preprocess_data(data):
    """Preprocess the data by splitting into features and target."""
    X = data.drop(columns=['Anomaly'])
    y = data['Anomaly']
    return (X, y)

def train_model(X_train, y_train):
    """Train a RandomForestClassifier and log the model using MLflow."""
    with mlflow.start_run():
        log_param('model', 'RandomForestClassifier')
        log_param('n_estimators', 100)
        log_param('max_depth', None)
        pipeline = Pipeline([('imputer', SimpleImputer(strategy='mean')), ('scaler', StandardScaler()), ('classifier', RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42))])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_train)
        accuracy = accuracy_score(y_train, y_pred)
        log_metric('train_accuracy', accuracy)
        log_model(pipeline, 'random_forest_model')
        return pipeline