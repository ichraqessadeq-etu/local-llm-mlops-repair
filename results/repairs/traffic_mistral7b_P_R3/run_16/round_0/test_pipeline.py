import unittest
from unittest.mock import patch
import pipeline
import pandas as pd

class TestPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = pd.read_csv("data/traffic.csv")

    def test_load_data(self):
        loaded_data = pipeline.load_data("data/traffic.csv")
        self.assertTrue(isinstance(loaded_data, pd.DataFrame))

    def test_preprocess_data(self):
        X, y = pipeline.preprocess_data(self.data)
        self.assertTrue(isinstance(X, pd.DataFrame))
        self.assertTrue(isinstance(y, pd.Series))

    @patch('pipeline.train_model')
    def test_train_model(self, mock_train_model):
        X_train, y_train = pipeline.preprocess_data(self.data)
        pipeline.train_model(X_train, y_train)
        mock_train_model.assert_called_once()

if __name__ == "__main__":
    unittest.main()
