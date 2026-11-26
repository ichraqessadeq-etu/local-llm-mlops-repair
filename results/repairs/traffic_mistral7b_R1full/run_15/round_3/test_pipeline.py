import unittest
import tempfile
from unittest.mock import patch
import pandas
import io
import os
import shutil

class TestPipeline(unittest.TestCase):
    @patch('pipeline.load_data')
    def test_load_data(self, mock_load_data):
        mock_data = {'temp': [1, 2, 3], 'rain_1h': [4, 5, 6]}
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.close()
        temp_file_path = temp_file.name
        pandas.DataFrame(mock_data).to_csv(temp_file_path)
        mock_load_data.return_value = pandas.read_csv(temp_file_path)
        result = pipeline.load_data(temp_file_path)
        self.assertIsInstance(result, pandas.DataFrame)
        os.remove(temp_file_path)

    @patch('pipeline.RandomForestClassifier')
    def test_train_model(self, mock_rf_classifier):
        mock_rf_classifier.return_value = mock_rf_classifier
        mock_data = {'temp': [1, 2, 3], 'rain_1h': [4, 5, 6]}
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.close()
        temp_file_path = temp_file.name
        pandas.DataFrame(mock_data).to_csv(temp_file_path)
        data = pipeline.load_data(temp_file_path)
        X, y = pipeline.preprocess_data(data)
        model = pipeline.train_model(X, y, random_state=42)
        self.assertIsInstance(model, mock_rf_classifier)
        os.remove(temp_file_path)

if __name__ == '__main__':
    unittest.main()
