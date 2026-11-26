import unittest
from pipeline import load_data, preprocess_data, train_model

class TestPipeline(unittest.TestCase):
    def setUp(self):
        self.data = load_data('data/traffic.csv')

    def test_load_data(self):
        self.assertTrue(self.data.columns.get_loc('Features') != -1)
        self.assertTrue(self.data.columns.get_loc('Congestion') != -1)

    def test_preprocess_data(self):
        X, y = preprocess_data(self.data)
        self.assertTrue(X.shape[1] > 0)
        self.assertTrue(y.shape[0] > 0)

    def test_train_model(self):
        X, y = preprocess_data(self.data)
        model = train_model(X, y)
        # Add assertions to check if the model is trained correctly
