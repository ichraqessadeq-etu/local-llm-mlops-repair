import pandas as pd
import unittest
from pipeline import load_data, preprocess_data, train_model

class TestPipeline(unittest.TestCase):
    def test_load_data(self):
        data = load_data('data/fraud.csv')
        self.assertIsInstance(data, pd.DataFrame)

    # Add more test cases for preprocess_data and train_model functions

if __name__ == '__main__':
    unittest.main()
