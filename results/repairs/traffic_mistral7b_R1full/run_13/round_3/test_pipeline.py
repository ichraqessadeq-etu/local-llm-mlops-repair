import unittest
import pandas as pd
from pipeline import load_data, preprocess_data, train_model

class TestPipeline(unittest.TestCase):
    def setUp(self):
        self.data = load_data('data/traffic.csv')
        self.X, self.y = preprocess_data(self.data)

    def test_train_model(self):
        train_model(self.X, self.y)
        # Add more tests as needed

    def test_load_data(self):
        data = load_data('data/traffic.csv')
        self.assertTrue(data.empty is False)
        self.assertTrue(len(data.columns) == 8)

if __name__ == '__main__':
    unittest.main()
