import unittest
from pipeline import load_data, preprocess_data, train_model
import pandas as pd
import numpy as np

class TestPipeline(unittest.TestCase):
    def setUp(self):
        self.data = load_data("data/traffic.csv")

    def test_load_data(self):
        self.assertIsInstance(self.data, pd.DataFrame)

    # Add more tests for preprocess_data and train_model functions

if __name__ == "__main__":
    unittest.main()
