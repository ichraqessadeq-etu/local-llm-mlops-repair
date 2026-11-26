import unittest
from pipeline import load_data, preprocess_data, train_model
from mlflow.tracking.artifact_utils import download_artifact

class TestPipeline(unittest.TestCase):
    def test_train(self):
        data = load_data("data/traffic.csv")
        X, y = preprocess_data(data)
        model = train_model(X, y)
        # Download the model for testing
        model_path = download_artifact("runs:/{run_id}/model", local_dest_dir="models")
        # Perform model testing here
        # ...
