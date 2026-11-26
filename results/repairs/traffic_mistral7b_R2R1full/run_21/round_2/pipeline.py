import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from mlflow.tracking import MlflowClient

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(data):
    return (data['Features'], data['Congestion'])

def train_model(X_train, y_train):
    with mlflow.start_run(run_name='RandomForest_Congestion_Detector'):
        try:
            model = RandomForestClassifier(random_state=42)
            model.fit(X_train, y_train)
            client = MlflowClient()
            artifact_path = client.create_artifact_repository(".")
            artifact_uri = artifact_path.uri
            artifact_local_path = artifact_path.local_path
            mlflow.pyfunc.save_model(model, artifact_local_path)
            mlflow.pyfunc.log_model(artifact_uri, "random_forest_congestion_detector")
            log_param('learning_rate', model.get_params()['max_depth'])
            return model
        except Exception as e:
            log_metric('training_error', 1)
            print(f'An error occurred during training: {e}')
