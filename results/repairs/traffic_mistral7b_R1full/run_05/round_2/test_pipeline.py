import unittest
import numpy as np
from pipeline import load_data, preprocess_data, train_model

class TestPipeline(unittest.TestCase):
    def setUp(self):
        self.data = load_data('data/traffic.csv')

    def test_load_data(self):
        data = load_data('data/traffic.csv')
        self.assertTrue(len(data) > 0)

    def test_preprocess_data(self):
        X, y = preprocess_data(self.data)
        self.assertTrue(len(X) > 0)
        self.assertTrue(len(y) > 0)

    def test_train_model(self):
        X_train, y_train = preprocess_data(self.data)
        model = train_model(X_train, y_train)
        # Add assertions to check if the model is trained correctly

if __name__ == "__main__":
    unittest.main()
