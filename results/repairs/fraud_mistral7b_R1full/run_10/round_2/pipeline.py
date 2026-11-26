import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from mlflow import log_artifact, pyfunc, tracking

def load_data(file_path):
    return pd.read_csv(file_path, index_col=0)

def preprocess_data(data):
    # Perform data preprocessing like feature scaling, handling missing values, etc.
    # For simplicity, I'll assume no preprocessing is needed.
    return data['Features'], data['Class']

def train_model(X_train, y_train):
    with mlflow.start_run():
        tracking.set_experiment("Fraud Detection")
        tracking.set_tag("version", "1.0.0")

        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)

        mlflow.pyfunc.log_model(
            pyfunc.DistributedModel.from_flask(model),
            artifact_path="model"
        )

        print(f"Model with version {mlflow.get_run_id()} has been saved.")
