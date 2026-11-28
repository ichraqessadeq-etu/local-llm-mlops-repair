import unittest
from pipeline import load_data, preprocess_data, train_model

class TestPipeline(unittest.TestCase):
    def test_load_data(self):
        data = load_data('data/traffic.csv')
        self.assertIsNotNone(data)

    def test_preprocess_data(self):
        data = load_data('data/traffic.csv')
        X, y = preprocess_data(data)
        self.assertIsNotNone(X)
        self.assertIsNotNone(y)

    def test_train_model(self):
        data = load_data('data/traffic.csv')
        X, y = preprocess_data(data)
        model = train_model(X, y)
        self.assertIsNotNone(model)

if __name__ == '__main__':
    unittest.main()
