import unittest
import numpy as np
from pipeline import load_data, preprocess_data, train_model

class TestPipeline(unittest.TestCase):
    def setUp(self):
        self.data = load_data("data/traffic.csv")

    def test_load_data(self):
        self.assertIsInstance(self.data, pd.DataFrame)

    def test_preprocess_data(self):
        X, y = preprocess_data(self.data)
        self.assertIsInstance(X, pd.DataFrame)
        self.assertIsInstance(y, pd.Series)

    def test_train_model(self):
        X, y = preprocess_data(self.data)
        train_model(X, y)
        # Add assertions to check the model's performance if needed

if __name__ == "__main__":
    unittest.main()
