import unittest
from pipeline import load_data, preprocess_data, train_model
import sklearn

class TestPipeline(unittest.TestCase):
    def setUp(self):
        self.data = load_data("data/fraud.csv")

    def test_load_data(self):
        self.assertIsInstance(self.data, pd.DataFrame)

    def test_preprocess_data(self):
        X, y = preprocess_data(self.data)
        self.assertIsInstance(X, pd.DataFrame)
        self.assertIsInstance(y, pd.Series)

    def test_train_model(self):
        X, y = preprocess_data(self.data)
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.3, random_state=42)
        train_model(X_train, y_train, random_state=42)
        # Add assertions to check the model's performance on the test set if needed
