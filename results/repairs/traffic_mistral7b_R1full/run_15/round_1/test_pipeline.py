import unittest
import tempfile
from unittest.mock import patch
import pipeline

class TestPipeline(unittest.TestCase):
    @patch('pipeline.pd.read_csv')
    def test_load_data(self, mock_read_csv):
        mock_data = {'temp': [1, 2, 3], 'rain_1h': [4, 5, 6]}
        mock_read_csv.return_value = pd.DataFrame(mock_data)
        result = pipeline.load_data(tempfile.mkdtemp())
        self.assertIsInstance(result, pd.DataFrame)

    @patch('pipeline.RandomForestClassifier')
    def test_train_model(self, mock_rf_classifier):
        mock_rf_classifier.return_value = mock_rf_classifier
        data = pipeline.load_data(tempfile.mkdtemp())
        X, y = pipeline.preprocess_data(data)
        model = pipeline.train_model(X, y, random_state=42)
        self.assertIsInstance(model, mock_rf_classifier)

if __name__ == '__main__':
    unittest.main()
