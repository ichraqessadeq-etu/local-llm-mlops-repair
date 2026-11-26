# test_pipeline.py
import unittest
import mlflow
from pipeline import load_data, preprocess_data, train_model

class TestPipeline(unittest.TestCase):
    def setUp(self):
        mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.autolog()

    def test_train(self):
        data = load_data("data/traffic.csv")
        X, y = preprocess_data(data)
        with mlflow.start_run():
            model = train_model(X, y)
            artifact_path = mlflow.get_artifact_uri("runs:/{run_id}/model")
            artifact = mlflow.artifacts.Artifact(artifact_path)
            artifact.download_local_artifact()
            # Perform model testing here
            # ...
