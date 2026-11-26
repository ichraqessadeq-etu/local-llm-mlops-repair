import unittest
from pipeline import load_data, preprocess_data, train_model
import pandas as pd
from sklearn.model_selection import train_test_split

class TestPipeline(unittest.TestCase):
    def setUp(self):
        self.data = load_data("data/fraud.csv")

    def test_load_data(self):
        self.assertIsInstance(self.data, pd.DataFrame)

    def test_preprocess_data(self):
        X, y = preprocess_data(self.data)
        self.assertIsInstance(X, pd.DataFrame)

    def test_train_model(self):
        X, y = preprocess_data(self.data)
        X_train, X_test, y_train, y_test = train_test_split(X.drop('Class', axis=1), y, test_size=0.2, random_state=42)
        model = train_model(X_train, y_train)
        # Add your custom assertions to check the model's performance

if __name__ == "__main__":
    unittest.main()
