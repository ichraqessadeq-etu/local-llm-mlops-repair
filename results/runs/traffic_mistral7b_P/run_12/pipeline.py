import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from mlflow import log_artifact, pyfunc, track

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def preprocess_data(data):
    # Perform any necessary preprocessing (e.g., normalization, encoding categorical variables)
    # For simplicity, I'll just drop the 'weather_main' column as it's not numeric
    data.drop('weather_main', axis=1, inplace=True)
    X = data.drop('Congestion', axis=1)
    y = data['Congestion']
    return X, y

def train_model(X_train, y_train):
    with mlflow.start_run():
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        mlflow.pyfunc.log_model(model, "model")
        print(f"Model with version {mlflow.get_run_id()} has been logged.")
