import unittest
import pandas as pd
from sklearn.model_selection import train_test_split
from pipeline import load_data, preprocess_data, train_model

class TestPipeline(unittest.TestCase):
    def setUp(self):
        self.data = load_data('data/fraud.csv')

    def test_load_data(self):
        self.assertIsInstance(self.data, pd.DataFrame)

    def test_preprocess_data(self):
        X, y = preprocess_data(self.data)
        self.assertIsInstance(X, pd.DataFrame)
        self.assertIsInstance(y, pd.Series)

    def test_train_model(self):
        X_train, y_train = train_test_split(self.data['Features'], self.data['Class'], test_size=0.2, random_state=42)
        model = train_model(X_train, y_train)
        # Add more assertions here to test the model's performance

    # Add more tests for edge cases and error handling
