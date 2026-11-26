import unittest
from pipeline import load_data, preprocess_data, train_model
from mlflow.tracking.artifact_utils import download_artifact
import numpy as np

class TestPipeline(unittest.TestCase):
    def setUp(self):
        self.data = load_data("data/traffic.csv")
        self.X, self.y = preprocess_data(self.data)

    def test_train(self):
        model = train_model(self.X, self.y)
        # Download the model for testing
        model_path = download_artifact("runs:/{run_id}/model", local_dest_dir="models")
        # Perform model testing here
        # ...

    def tearDown(self):
        pass
