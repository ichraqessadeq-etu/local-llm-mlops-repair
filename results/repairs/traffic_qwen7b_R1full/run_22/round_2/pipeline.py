# pipeline.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
import mlflow
import mlflow.sklearn

def load_data(file_path):
    """Load data from a CSV file."""
    return pd.read_csv(file_path)

def preprocess_data(data):
    """Preprocess the data."""
    # Assuming 'Congestion' is the target column
    X = data.drop(columns=['Congestion'])
    y = data['Congestion']
    
    # One-hot encode the 'weather_main' column
    encoder = OneHotEncoder(sparse=False)
    weather_encoded = encoder.fit_transform(X[['weather_main']])
    weather_encoded_df = pd.DataFrame(weather_encoded, columns=encoder.get_feature_names_out(['weather_main']))
    X = X.drop(columns=['weather_main']).join(weather_encoded_df)
    
    return X, y

def train_model(X_train, y_train):
    """Train a RandomForestClassifier."""
    with mlflow.start_run():
        mlflow.log_param("model", "RandomForestClassifier")
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        mlflow.sklearn.log_model(model, "model")
        y_pred = model.predict(X_train)
        accuracy = accuracy_score(y_train, y_pred)
        mlflow.log_metric("train_accuracy", accuracy)
        return model
