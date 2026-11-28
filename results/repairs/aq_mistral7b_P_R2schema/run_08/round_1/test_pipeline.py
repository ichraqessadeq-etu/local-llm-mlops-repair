from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import unittest
from unittest.mock import patch
from pipeline import load_data, preprocess_data, train_model

class TestPipeline(unittest.TestCase):

    @patch('pipeline.pd.read_csv')
    def test_load_data(self, mock_read_csv):
        mock_data = {'Features': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], 'Anomaly': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]}
        mock_read_csv.return_value = mock_data
        result = load_data('data/air_quality.csv')
        self.assertEqual(result, mock_data)

    @patch('pipeline.pd.read_csv')
    def test_preprocess_data(self, mock_read_csv):
        mock_data = {'Features': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], 'Anomaly': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]}
        mock_read_csv.return_value = mock_data
        X, y = preprocess_data(mock_data)
        self.assertIsInstance(X, pd.DataFrame)
        self.assertIsInstance(y, pd.Series)

    @patch('pipeline.pd.read_csv')
    def test_train_model(self, mock_read_csv):
        mock_data = {'Features': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], 'Anomaly': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]}
        mock_read_csv.return_value = mock_data
        X, y = (mock_data.iloc[:, :-1], mock_data['Anomaly'])
        mock_model = RandomForestClassifier(random_state=42)
        mock_model.fit(X, y)
        train_model(X, y)