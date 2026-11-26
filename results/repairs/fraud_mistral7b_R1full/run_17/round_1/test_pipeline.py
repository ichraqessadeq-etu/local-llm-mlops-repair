import unittest
from pipeline import load_data, preprocess_data, train_model
import pandas as pd

class TestPipeline(unittest.TestCase):
    def test_load_data(self):
        data = load_data("data/fraud.csv")
        self.assertIsInstance(data, pd.DataFrame)

    def test_preprocess_data(self):
        data, _ = preprocess_data(load_data("data/fraud.csv"))
        self.assertIsInstance(data, tuple)
        self.assertIsInstance(data[0], pd.DataFrame)
        self.assertIsInstance(data[1], pd.Series)

    def test_train_model(self):
        X, y = preprocess_data(load_data("data/fraud.csv"))
        model = train_model(X_train=X, y_train=y)
        self.assertIsInstance(model, RandomForestClassifier)

if __name__ == "__main__":
    unittest.main()
