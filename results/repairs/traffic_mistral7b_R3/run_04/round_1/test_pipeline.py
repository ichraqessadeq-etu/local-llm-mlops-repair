import pandas as pd
import unittest
from pipeline import load_data, preprocess_data, train_model

class TestPipeline(unittest.TestCase):
    def test_load_data(self):
        data = load_data("data/traffic.csv")
        self.assertIsInstance(data, pd.DataFrame)

    def test_preprocess_data(self):
        data, _ = preprocess_data(load_data("data/traffic.csv"))
        self.assertIsInstance(data, pd.DataFrame)

    # Add more tests for train_model and error handling as needed
