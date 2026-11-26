import unittest
from pipeline import load_data, preprocess_data, train_model

class TestPipeline(unittest.TestCase):
    def test_load_data(self):
        data = load_data('data/traffic.csv')
        self.assertTrue(data.columns.get_loc('Features') != -1)
        self.assertTrue(data.columns.get_loc('Congestion') != -1)

    def test_preprocess_data(self):
        data = load_data('data/traffic.csv')
        X, y = preprocess_data(data)
        self.assertTrue(X.shape[1] > 0)
        self.assertTrue(y.shape[0] > 0)

    def test_train_model(self):
        data = load_data('data/traffic.csv')
        X, y = preprocess_data(data)
        model = train_model(X, y)
        # Add assertions to check if the model is trained correctly
