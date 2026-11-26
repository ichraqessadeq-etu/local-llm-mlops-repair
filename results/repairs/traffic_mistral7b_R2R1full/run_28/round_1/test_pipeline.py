# test_pipeline.py
import unittest
from pipeline import load_data, preprocess_data, train_model

class TestPipeline(unittest.TestCase):
    def test_load_data(self):
        # Test loading data from the correct file
        data = load_data('data/traffic.csv')
        # Add assertions to check if the data is loaded correctly

    def test_preprocess_data(self):
        # Test preprocessing data correctly
        data = load_data('data/traffic.csv')
        X, y = preprocess_data(data)
        # Add assertions to check if X and y are as expected

    def test_train_model(self):
        # Test training the model and logging it correctly
        data = load_data('data/traffic.csv')
        X, y = preprocess_data(data)
        model = train_model(X, y)
        # Add assertions to check if the model is trained correctly and logged

if __name__ == "__main__":
    unittest.main()
