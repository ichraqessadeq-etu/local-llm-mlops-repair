import unittest
import tempfile
from unittest.mock import patch
from pipeline import load_data, preprocess_data, train_model

class TestPipeline(unittest.TestCase):
    def setUp(self):
        self.data = load_data('data/traffic.csv')

    def test_load_data(self):
        self.assertIsInstance(self.data, pd.DataFrame)

    def test_preprocess_data(self):
        X, y = preprocess_data(self.data)
        self.assertIsInstance(X, np.ndarray)
        self.assertIsInstance(y, np.ndarray)

    @patch('pipeline.RandomForestClassifier', new_callable=lambda *args, **kwargs: unittest.mock.Mock())
    def test_train_model(self, mock_rf):
        X_train, y_train = preprocess_data(self.data)
        train_model(X_train, y_train)
        mock_rf.assert_called_once_with(random_state=42)

if __name__ == "__main__":
    unittest.main()
