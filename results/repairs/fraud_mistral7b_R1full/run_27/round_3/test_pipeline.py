import unittest
import numpy as np
from pipeline import load_data, preprocess_data, train_model
from sklearn.model_selection import train_test_split

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
        X_train, y_train = train_test_split(self.data.values, self.data['Class'], test_size=0.2, random_state=42)
        X_train = pd.DataFrame(X_train, columns=self.data.columns[:-1])
        y_train = pd.Series(y_train)
        train_model(X_train, y_train)
        # Add assertions if necessary

    # Add more test cases for error handling
