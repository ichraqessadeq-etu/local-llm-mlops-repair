from sklearn.model_selection import train_test_split
import unittest
from pipeline import load_data, preprocess_data, train_model
import pandas as pd

class TestPipeline(unittest.TestCase):
    def setUp(self):
        self.data = load_data('data/air_quality.csv')
        self.X, self.y = preprocess_data(self.data)

    def test_train_model(self):
        train_X, test_X, train_y, test_y = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        train_model(train_X, train_y)

        # You can add more tests to check the model's performance on the test set

if __name__ == '__main__':
    unittest.main()
