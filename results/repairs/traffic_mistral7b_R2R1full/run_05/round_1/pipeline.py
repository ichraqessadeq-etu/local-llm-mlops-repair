# pipeline.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.sklearn

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(data):
    data.columns = ['Features', 'Congestion']  # Added this line to fix the column name
    return (data['Features'], data['Congestion'])

def train_model(X_train, y_train):
    with mlflow.start_run():
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        mlflow.sklearn.log_model(model, 'random-forest')
    return model
