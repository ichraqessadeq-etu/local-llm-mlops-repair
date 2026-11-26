import unittest
from unittest.mock import patch
from pipeline import load_data, preprocess_data, train_model, save_model

class TestPipeline(unittest.TestCase):
    @patch('pipeline.pd.read_csv')
    def test_load_data(self, mock_read_csv):
        mock_data = pd.DataFrame()
        mock_read_csv.return_value = mock_data
        result = load_data('data/fraud.csv')
        self.assertEqual(result, mock_data)

    @patch('pipeline.pd.DataFrame')
    def test_preprocess_data(self, mock_df):
        mock_df_return_value = mock_df(columns=['V1', 'V2'])
        mock_df.return_value = mock_df_return_value
        data = load_data('data/fraud.csv')
        X, y = preprocess_data(data)
        self.assertTrue(isinstance(X, pd.DataFrame))
        self.assertTrue(isinstance(y, pd.Series))

    @patch('pipeline.RandomForestClassifier')
    def test_train_model(self, mock_rf):
        mock_rf.return_value = object()
        X, y = preprocess_data(load_data('data/fraud.csv'))
        model = train_model(X, y)
        self.assertIsInstance(model, object)

    @patch('pipeline.save_model')
    def test_save_model(self, mock_save_model):
        model = object()
        save_model(model, 'model')
        mock_save_model.assert_called_once_with(model, 'model')

if __name__ == '__main__':
    unittest.main()
