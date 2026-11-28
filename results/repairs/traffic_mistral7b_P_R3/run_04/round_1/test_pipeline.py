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

    @patch('pipeline.pd.DataFrame')
    def test_preprocess_data(self, mock_DataFrame):
        mock_df = pd.DataFrame()
        mock_X, mock_y = mock_df, mock_df['Congestion']
        mock_DataFrame.return_value = mock_df
        result = preprocess_data(mock_df)
        self.assertEqual(result[0], mock_df)
        self.assertEqual(result[1], mock_y)

if __name__ == '__main__':
    unittest.main()
