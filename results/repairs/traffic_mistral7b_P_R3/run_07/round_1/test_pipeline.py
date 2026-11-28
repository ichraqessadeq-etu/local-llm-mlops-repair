import pandas as pd
import unittest
from unittest.mock import patch
from pipeline import load_data, preprocess_data, train_model

class TestPipeline(unittest.TestCase):
    @patch('pipeline.pd.read_csv')
    def test_load_data(self, mock_read_csv):
        mock_data = pd.DataFrame()
        mock_read_csv.return_value = mock_data
        result = load_data('data/traffic.csv')
        self.assertEqual(result, mock_data)

    @patch('pipeline.pd.read_csv')
    def test_preprocess_data(self, mock_read_csv):
        mock_data = pd.DataFrame()
        mock_data['Congestion'] = [0, 1, 0, 1]
        mock_read_csv.return_value = mock_data
        result, y = preprocess_data(mock_data)
        self.assertEqual(len(result), len(mock_data))
        self.assertEqual(len(y), len(mock_data))

if __name__ == '__main__':
    unittest.main()
