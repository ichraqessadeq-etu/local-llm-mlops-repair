import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from mlflow.tracking import MlflowClient
from mlflow.entities import Run

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(data):
    # Preprocessing steps like filling NaN, encoding categorical variables, etc.
    # For simplicity, I'm assuming that the data is already preprocessed.
    return data['Features'], data['Congestion']

def train_model(X_train, y_train):
    with MlflowClient() as client:
        run_id = client.create_run(experiment_id="experiment-id")
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        artifact_path = f"models/{run_id}/model"
        client.log_artifact(artifact_path, "model")
        client.log_param("num_trees", model.n_estimators)
        run = Run.get_run_by_id(run_id)
        print(f"Run ID: {run.info.run_id}, Run Status: {run.info.status}")
