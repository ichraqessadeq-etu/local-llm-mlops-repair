import unittest
import tempfile
import pandas as pd
from pipeline import load_data, preprocess_data, train_model

class TestPipeline(unittest.TestCase):
    def test_load_data(self):
        data = load_data("data/traffic.csv")
        self.assertIsInstance(data, pd.DataFrame)

    def test_preprocess_data(self):
        data = load_data("data/traffic.csv")
        X, y = preprocess_data(data)
        self.assertIsInstance(X, pd.DataFrame)
        self.assertIsInstance(y, pd.Series)

    def test_train_model(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data = load_data("data/traffic.csv")
            X, y = preprocess_data(data)
            model = train_model(X, y)
            self.assertIsInstance(model, RandomForestClassifier)

if __name__ == "__main__":
    unittest.main()
