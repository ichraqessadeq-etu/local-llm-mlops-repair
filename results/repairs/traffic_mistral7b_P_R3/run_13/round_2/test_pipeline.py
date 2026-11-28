import unittest
from unittest.mock import patch
import pipeline
import pandas as pd

class TestPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = pd.read_csv("data/traffic.csv")
        cls.X, cls.y = pipeline.preprocess_data(cls.data)

    def test_train_model(self):
        with patch('pipeline.load_data') as mock_load_data:
            mock_load_data.return_value = self.data
            model = pipeline.train_model(self.X, self.y)
            mock_load_data.assert_called_once_with("data/traffic.csv")

if __name__ == "__main__":
    unittest.main()
