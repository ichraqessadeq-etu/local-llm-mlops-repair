# test_pipeline.py
import pandas as pd
import unittest
import numpy as np
from pipeline import load_data, preprocess_data, train_model

class TestPipeline(unittest.TestCase):
    def setUp(self):
        data = load_data("data/traffic.csv")
        self.X, self.y = preprocess_data(data)
        self.X_train, self.y_train, self.X_test, self.y_test = train_test_split(self.X, self.y, test_size=0.3, random_state=42)

    def test_load_data(self):
        data = load_data("data/traffic.csv")
        self.assertIsInstance(data, pd.DataFrame)

    def test_train_model(self):
        model = train_model(self.X_train, self.y_train)
        # Add assertions to check model properties or performance metrics

    def test_model_performance(self):
        model = train_model(self.X_train, self.y_train)
        self.assertGreater(model.score(self.X_train, self.y_train), 0.5)
        self.assertGreater(model.score(self.X_test, self.y_test), 0.5)

if __name__ == "__main__":
    unittest.main()
