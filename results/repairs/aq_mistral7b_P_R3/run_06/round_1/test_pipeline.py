import unittest
from pipeline import load_data, preprocess_data

class TestPipeline(unittest.TestCase):
    def setUp(self):
        self.data = load_data('data/air_quality.csv')
        self.X, self.y = preprocess_data(self.data)

    def test_load_data(self):
        self.assertTrue(len(self.data) > 0)

    def test_preprocess_data(self):
        X, y = preprocess_data(self.data)
        self.assertTrue(len(X) > 0)
        self.assertTrue(len(y) > 0)

if __name__ == "__main__":
    unittest.main()
