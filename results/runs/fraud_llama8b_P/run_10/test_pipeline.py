import unittest
from pipeline import load_data, preprocess_data, train_model

class TestPipeline(unittest.TestCase):
    def test_load_data(self):
        data = load_data('data/fraud.csv')
        self.assertIsNotNone(data)
    
    def test_preprocess_data(self):
        data = load_data('data/fraud.csv')
        X_train, X_test, y_train, y_test = preprocess_data(data)
        self.assertIsNotNone(X_train)
        self.assertIsNotNone(X_test)
        self.assertIsNotNone(y_train)
        self.assertIsNotNone(y_test)
    
    def test_train_model(self):
        data = load_data('data/fraud.csv')
        X_train, X_test, y_train, y_test = preprocess_data(data)
        model = train_model(X_train, y_train)
        self.assertIsNotNone(model)

if __name__ == '__main__':
    unittest.main()
