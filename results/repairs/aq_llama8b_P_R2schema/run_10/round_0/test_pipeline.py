import unittest
from pipeline import load_data, preprocess_data, train_model

class TestPipeline(unittest.TestCase):
    def test_load_data(self):
        data = load_data('data/air_quality.csv')
        self.assertIsNotNone(data)

    def test_preprocess_data(self):
        data = load_data('data/air_quality.csv')
        X, y = preprocess_data(data)
        self.assertIsNotNone(X)
        self.assertIsNotNone(y)

    def test_train_model(self):
        data = load_data('data/air_quality.csv')
        X, y = preprocess_data(data)
        model = train_model(X, y)
        self.assertIsNotNone(model)
