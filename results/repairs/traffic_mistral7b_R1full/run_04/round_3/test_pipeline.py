import unittest
import numpy as np
from pipeline import load_data, preprocess_data, train_model

class TestPipeline(unittest.TestCase):
    def test_load_data(self):
        data = load_data("data/traffic.csv")
        self.assertIsInstance(data, pd.DataFrame)

    def test_preprocess_data(self):
        data, _ = preprocess_data(load_data("data/traffic.csv"))
        self.assertIsInstance(data, pd.DataFrame)

    def test_train_model(self):
        data = load_data("data/traffic.csv")
        X, y = preprocess_data(data)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        model = train_model(X_train, y_train)
        # Add more tests for model predictions and error handling as needed
