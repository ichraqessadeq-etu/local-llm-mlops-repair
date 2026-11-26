import pandas as pd
import unittest
from pipeline import load_data, preprocess_data, train_model

class TestPipeline(unittest.TestCase):
    def test_load_data(self):
        data = load_data('data/fraud.csv')
        self.assertIsInstance(data, pd.DataFrame)

    def test_preprocess_data(self):
        data = load_data('data/fraud.csv')
        X, y = preprocess_data(data)
        self.assertIsInstance(X, pd.DataFrame)
        self.assertIsInstance(y, pd.Series)

    # Add more test cases for train_model and error handling
