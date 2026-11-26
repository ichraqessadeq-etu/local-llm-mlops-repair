import pandas as pd
import unittest
from unittest.mock import patch
import pipeline

class TestPipeline(unittest.TestCase):
    @patch('pipeline.pd.read_csv')
    def test_load_data(self, mock_read_csv):
        mock_data = {'temp': [1, 2, 3], 'rain_1h': [4, 5, 6]}
        mock_read_csv.return_value = pd.DataFrame(mock_data)
        result = pipeline.load_data('data/traffic.csv')
        self.assertIsInstance(result, pd.DataFrame)

    # Add more test cases for preprocess_data and train_model functions

if __name__ == '__main__':
    unittest.main()
