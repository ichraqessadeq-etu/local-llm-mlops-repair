import unittest
import os
import tempfile
from unittest.mock import patch
from pipeline import load_data, preprocess_data, train_model

class TestPipeline(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_load_data(self):
        data = load_data(os.path.join(self.temp_dir.name, "air_quality.csv"))
        self.assertIsInstance(data, pd.DataFrame)

    @patch('pipeline.RandomForestClassifier', new_callable=property)
    def test_train_model(self, mock_rf):
        mock_rf.return_value.fit.return_value = None
        X, y = preprocess_data(load_data(os.path.join(self.temp_dir.name, "air_quality.csv")))
        train_model(X, y)
        mock_rf.assert_called_once_with(random_state=42)

if __name__ == "__main__":
    unittest.main()
