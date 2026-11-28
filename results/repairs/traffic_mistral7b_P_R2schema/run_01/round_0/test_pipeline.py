import unittest
from unittest.mock import patch
from pipeline import load_data, preprocess_data, train_model

class TestPipeline(unittest.TestCase):
    @patch('pipeline.pd.read_csv')
    def test_load_data(self, mock_read_csv):
        mock_data = {'features': [1, 2, 3], 'Congestion': [0, 1, 0]}
        mock_read_csv.return_value = mock_data
        result = load_data('data/traffic.csv')
        expected = mock_data
        self.assertEqual(result, expected)

    @patch('pipeline.pd.DataFrame')
    def test_preprocess_data(self, mock_dataframe):
        mock_data = {'features': [1, 2, 3], 'Congestion': [0, 1, 0]}
        mock_dataframe.return_value = mock_data
        result, _ = preprocess_data(mock_data)
        expected_X = mock_data[['temp', 'rain_1h', 'snow_1h', 'clouds_all', 'weather_main', 'IsHoliday', 'Hour', 'DayOfWeek', 'Month']]
        expected_y = mock_data['Congestion']
        self.assertEqual(result[0], expected_X)
        self.assertEqual(result[1], expected_y)

if __name__ == '__main__':
    unittest.main()
