import unittest
import pandas as pd
from pipeline import load_data, preprocess_data, train_model

class TestPipeline(unittest.TestCase):
    def test_load_data(self):
        data = load_data('data/traffic.csv')
        self.assertIsInstance(data, pd.DataFrame)

    def test_preprocess_data(self):
        data = pd.DataFrame({'Features': [1, 2, 3], 'Congestion': [0, 1, 0]})
        X, y = preprocess_data(data)
        self.assertIsInstance(X, pd.DataFrame)
        self.assertIsInstance(y, pd.Series)

    def test_train_model(self):
        data = pd.DataFrame({'Features': [[1, 2, 3, 4, 5, 6, 7, 8, 9, 0]],
                              ['Congestion': [0]]})
        X_train, y_train = preprocess_data(data)
        train_model(X_train, y_train)
        # You can add more tests to check the model's performance

if __name__ == '__main__':
    unittest.main()
