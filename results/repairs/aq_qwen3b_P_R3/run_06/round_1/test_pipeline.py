from sklearn.ensemble import RandomForestClassifier
import unittest
import pandas as pd
from pipeline import load_data, preprocess_data, train_model, evaluate_model

class TestPipeline(unittest.TestCase):
    def setUp(self):
        self.file_path = 'data/air_quality.csv'

    def test_load_data(self):
        data = load_data(self.file_path)
        self.assertIsInstance(data, pd.DataFrame)

    def test_preprocess_data(self):
        data = pd.DataFrame({
            'PT08.S1(CO)': [1, 2, 3],
            'Anomaly': [0, 1, 0]
        })
        X, y = preprocess_data(data)
        self.assertEqual(X.shape, (3, 1))
        self.assertEqual(y.shape, (3,))

    def test_train_model(self):
        data = pd.DataFrame({
            'PT08.S1(CO)': [1, 2, 3],
            'Anomaly': [0, 1, 0]
        })
        X, y = preprocess_data(data)
        model = train_model(X, y)
        self.assertIsInstance(model, RandomForestClassifier)

    def test_evaluate_model(self):
        data = pd.DataFrame({
            'PT08.S1(CO)': [1, 2, 3],
            'Anomaly': [0, 1, 0]
        })
        X, y = preprocess_data(data)
        model = train_model(X, y)
        accuracy = evaluate_model(model, X, y)
        self.assertIsInstance(accuracy, float)

if __name__ == '__main__':
    unittest.main()
