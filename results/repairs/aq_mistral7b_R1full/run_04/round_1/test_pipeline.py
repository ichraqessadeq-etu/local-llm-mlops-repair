import unittest
from pipeline import load_data, preprocess_data, train_model

class TestPipeline(unittest.TestCase):
    def setUp(self):
        self.data = load_data('data/air_quality.csv')

    def test_load_data(self):
        # Add your assertions here

    def test_preprocess_data(self):
        X, y = preprocess_data(self.data)
        # Add your assertions here

    def test_train_model(self):
        X, y = preprocess_data(self.data)
        model = train_model(X, y)
        # Add your assertions here

    def tearDown(self):
        del self.data

if __name__ == '__main__':
    unittest.main()
