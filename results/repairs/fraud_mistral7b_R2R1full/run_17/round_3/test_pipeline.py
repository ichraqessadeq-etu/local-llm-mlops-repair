from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import unittest
from pipeline import load_data, preprocess_data, train_model, save_model

class TestPipeline(unittest.TestCase):
    def setUp(self):
        self.data = load_data("data/fraud.csv")
        self.X, self.y = preprocess_data(self.data)

    def test_preprocess_data(self):
        X, y = preprocess_data(self.data)
        self.assertIsInstance(X, pd.DataFrame)
        self.assertIsInstance(y, pd.Series)

    def test_train_model(self):
        model = train_model(self.X, self.y)
        self.assertIsInstance(model, RandomForestClassifier)
        save_model(model, 'test_model')

if __name__ == "__main__":
    unittest.main()
