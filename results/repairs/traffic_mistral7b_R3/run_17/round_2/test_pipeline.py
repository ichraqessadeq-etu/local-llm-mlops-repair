from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import unittest
from pipeline import load_data, preprocess_data, train_model

class TestPipeline(unittest.TestCase):
    def test_load_data(self):
        data = load_data('data/traffic.csv')
        self.assertIsInstance(data, pd.DataFrame)

    def test_preprocess_data(self):
        data = load_data('data/traffic.csv')
        X, y = preprocess_data(data)
        self.assertIsInstance(X, pd.DataFrame)
        self.assertIsInstance(y, pd.Series)

    def test_train_model(self):
        data = load_data('data/traffic.csv')
        X, y = preprocess_data(data)
        model = train_model(X, y)
        self.assertIsInstance(model, RandomForestClassifier)

if __name__ == "__main__":
    unittest.main()
