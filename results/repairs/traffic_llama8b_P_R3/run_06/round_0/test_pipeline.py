import unittest
from pipeline import load_data, preprocess_data, train_model

class TestPipeline(unittest.TestCase):
    def test_load_data(self):
        data = load_data('data/traffic.csv')
        self.assertEqual(data.shape, (1000, 9))  # assuming 1000 rows and 9 columns

    def test_preprocess_data(self):
        data = load_data('data/traffic.csv')
        X_train, X_test, y_train, y_test = preprocess_data(data)
        self.assertEqual(X_train.shape, (800, 8))  # assuming 800 rows and 8 features
        self.assertEqual(y_train.shape, (800,))  # assuming 800 rows

    def test_train_model(self):
        data = load_data('data/traffic.csv')
        X_train, X_test, y_train, y_test = preprocess_data(data)
        model = train_model(X_train, y_train)
        self.assertIsInstance(model, RandomForestClassifier)

if __name__ == '__main__':
    unittest.main()
