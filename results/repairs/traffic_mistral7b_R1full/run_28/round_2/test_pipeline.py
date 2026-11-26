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
        self.assertTrue(len(X.columns) == 9)
        self.assertTrue(len(y) > 0)

    def test_train_model(self):
        X, y = preprocess_data(self.data)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        model = train_model(X_train, y_train)
        # Add assertions to check if the model is trained correctly and logged
        self.assertTrue(hasattr(model, 'fit'))
        self.assertTrue(hasattr(model, 'predict'))

if __name__ == "__main__":
    unittest.main()
