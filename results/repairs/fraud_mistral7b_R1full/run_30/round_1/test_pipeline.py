import unittest
from pipeline import load_data, preprocess_data, train_model
import pandas
import sklearn.model_selection

class TestPipeline(unittest.TestCase):
    def test_load_data(self):
        data = load_data('data/fraud.csv')
        self.assertIsInstance(data, pandas.DataFrame)

    def test_preprocess_data(self):
        data = load_data('data/fraud.csv')
        X, y = preprocess_data(data)
        self.assertIsInstance(X, pandas.DataFrame)
        self.assertIsInstance(y, pandas.Series)

    def test_train_model(self):
        data = load_data('data/fraud.csv')
        X, y = preprocess_data(data)
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.3, random_state=42)
        train_model(X_train, y_train)
