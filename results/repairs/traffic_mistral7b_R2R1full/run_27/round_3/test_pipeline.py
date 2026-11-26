# test_pipeline.py
import unittest
from pipeline import load_data, preprocess_data, train_model

class TestPipeline(unittest.TestCase):
    def setUp(self):
        self.data = load_data('data/traffic.csv')
        self.X, self.y = preprocess_data(self.data)

    def test_train_model(self):
        model = train_model(self.X, self.y)
        self.assertTrue(len(model.feature_importances_) > 0)
        self.assertTrue(model.score(self.X, self.y) > 0.5)

if __name__ == '__main__':
    unittest.main()
