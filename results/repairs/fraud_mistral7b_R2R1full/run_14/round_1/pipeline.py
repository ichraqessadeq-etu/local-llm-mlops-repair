# pipeline.py
import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(data):
    data.dropna(inplace=True)
    X = data.drop('Class', axis=1)
    y = data['Class']
    return (X, y)

def train_model(X_train, y_train):
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    mlflow.log_params({'feature_importances': model.feature_importances_})
    mlflow.sklearn.log_model(model, 'model')
