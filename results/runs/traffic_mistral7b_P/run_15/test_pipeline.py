import unittest
from unittest.mock import patch
from pipeline import load_data, preprocess_data, train_model

class TestPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = load_data("data/traffic.csv")
        cls.X, cls.y = preprocess_data(cls.data)

    def test_train_model(self):
        with patch('pipeline.train_model') as mock_train_model:
            train_model(self.X_train, self.y_train)
            mock_train_model.assert_called_once()

if __name__ == "__main__":
    unittest.main()
