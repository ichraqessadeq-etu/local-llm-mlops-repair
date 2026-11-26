import unittest
import mock
from unittest.mock import patch
from pipeline import load_data, preprocess_data, train_model

class TestPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = load_data('data/fraud.csv')

    @patch('time.time', return_value=42)
    def test_train_model(self, mock_time):
        X, y = preprocess_data(self.data)
        train_model(X_train=X, y_train=y)

    # Add more test cases for preprocess_data and other functions

if __name__ == '__main__':
    unittest.main()
