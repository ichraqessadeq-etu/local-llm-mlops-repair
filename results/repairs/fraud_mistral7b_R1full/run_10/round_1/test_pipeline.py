import unittest
import numpy as np
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
        X, y = preprocess_data(self.data)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        train_model(X_train, y_train)

    def test_reproducibility(self):
        np.random.seed(42)
        X, y = preprocess_data(self.data)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        model = train_model(X_train, y_train)

        np.random.seed(42)
        model2 = train_model(X_train, y_train)

        self.assertTrue(np.array_equal(model.predict(X_test), model2.predict(X_test)))
