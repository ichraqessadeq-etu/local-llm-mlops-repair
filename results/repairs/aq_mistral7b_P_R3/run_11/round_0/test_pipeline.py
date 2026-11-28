import unittest
from unittest.mock import patch
from pipeline import load_data, preprocess_data, train_model

class TestPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = load_data("data/air_quality.csv")
        cls.X, cls.y = preprocess_data(cls.data)

    def test_train_model(self):
        with patch("pipeline.pd.read_csv") as mock_read_csv:
            mock_read_csv.return_value = self.data
            model = train_model(self.X, self.y)
            # Add your assertions here

if __name__ == "__main__":
    unittest.main()
