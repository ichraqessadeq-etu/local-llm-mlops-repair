import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from mlflow import log_artifact, pyfunc, tracking

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(data):
    # Perform any necessary preprocessing such as normalization, encoding, etc.
    # For simplicity, I'll just return the data as is.
    return data['Features'], data['Class']

def train_model(X_train, y_train, X_test=None, y_test=None, random_state=42):
    with mlflow.start_run():
        model = RandomForestClassifier(random_state=random_state)
        model.fit(X_train, y_train)
        pyfunc.log_model(model, "model", registered_model_name="RandomForestFraudDetector")
        tracking.set_global_context(run_id=mlflow.active_run().info.run_id)
        track("Model training time")
    if X_test is not None and y_test is not None:
        model.score(X_test, y_test)
    return model
