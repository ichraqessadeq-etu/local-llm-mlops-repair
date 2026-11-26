import unittest
from pipeline import load_data, preprocess_data, train_model, main

class TestPipeline(unittest.TestCase):
    def test_load_data(self):
        try:
            data = load_data('data/air_quality.csv')
            self.assertIsNotNone(data)
        except Exception as e:
            self.fail(f"Failed to load data: {e}")

    def test_preprocess_data(self):
        try:
            data = load_data('data/air_quality.csv')
            X_train, X_test, y_train, y_test = preprocess_data(data)
            self.assertIsNotNone(X_train)
            self.assertIsNotNone(X_test)
            self.assertIsNotNone(y_train)
            self.assertIsNotNone(y_test)
        except Exception as e:
            self.fail(f"Failed to preprocess data: {e}")

    def test_train_model(self):
        try:
            data = load_data('data/air_quality.csv')
            X_train, X_test, y_train, y_test = preprocess_data(data)
            model = train_model(X_train, y_train)
            self.assertIsNotNone(model)
        except Exception as e:
            self.fail(f"Failed to train model: {e}")

if __name__ == "__main__":
    unittest.main()
