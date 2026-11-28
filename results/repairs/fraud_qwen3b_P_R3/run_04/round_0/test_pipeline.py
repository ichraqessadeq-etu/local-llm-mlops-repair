import unittest
from pipeline import load_data, preprocess_data, train_model, log_model

class TestPipeline(unittest.TestCase):
    def setUp(self):
        self.file_path = 'data/fraud.csv'
        self.data = load_data(self.file_path)

    def test_load_data(self):
        self.assertIsNotNone(self.data)

    def test_preprocess_data(self):
        X_train, X_test, y_train, y_test = preprocess_data(self.data)
        self.assertIsNotNone(X_train)
        self.assertIsNotNone(X_test)
        self.assertIsNotNone(y_train)
        self.assertIsNotNone(y_test)

    def test_train_model(self):
        X_train, X_test, y_train, y_test = preprocess_data(self.data)
        model = train_model(X_train, y_train)
        self.assertIsNotNone(model)

    def test_log_model(self):
        X_train, X_test, y_train, y_test = preprocess_data(self.data)
        model = train_model(X_train, y_train)
        log_model(model, 'fraud_detection', 'models')

if __name__ == '__main__':
    unittest.main()
