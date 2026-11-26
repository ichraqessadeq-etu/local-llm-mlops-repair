import unittest
from pipeline import load_data, preprocess_data, train_model
import numpy as np

class TestPipeline(unittest.TestCase):
    def setUp(self):
        self.data = load_data('data/traffic.csv')

    def test_load_data(self):
        # Test loading data from the correct file
        self.assertTrue(len(self.data) > 0)

    def test_preprocess_data(self):
        X, y = preprocess_data(self.data)
        # Add assertions to check if X and y are as expected
        self.assertTrue(len(X) > 0)
        self.assertTrue(len(y) > 0)

    def test_train_model(self):
        X, y = preprocess_data(self.data)
        model = train_model(X, y)
        # Add assertions to check if the model is trained correctly and logged
        self.assertTrue(hasattr(model, 'fit'))
        self.assertTrue(hasattr(model, 'predict'))

if __name__ == "__main__":
    unittest.main()
