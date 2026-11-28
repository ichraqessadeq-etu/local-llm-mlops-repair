import unittest
from unittest.mock import patch
import pipeline

class TestPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = load_data("data/air_quality.csv")
        cls.X, cls.y = preprocess_data(cls.data)

    def test_train_model(self):
        with patch('pipeline.mlflow.pyfunc.track') as mock_track:
            mock_track.return_value = None
            model = train_model(self.X_train, self.y_train)
            mock_track.assert_called_once()
