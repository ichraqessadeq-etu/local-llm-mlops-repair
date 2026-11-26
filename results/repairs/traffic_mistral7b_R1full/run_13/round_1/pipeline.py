import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from mlflow import log_artifact, pyfunc, run_local

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(data):
    # Preprocessing steps like filling NaN, encoding categorical variables, etc.
    # For simplicity, I'm assuming that the data is already preprocessed.
    return data['Features'].values, data['Congestion'].values

def train_model(X_train, y_train):
    with mlflow.start_run():
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        mlflow.pyfunc.log_model(model, "model")
        # Uncomment this line if you want to track the model's score
        # mlflow.pyfunc.log_metric("score", model.score(X_train, y_train))
