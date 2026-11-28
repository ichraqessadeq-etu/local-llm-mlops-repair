import unittest
from pipeline import load_data, preprocess_data, train_model, evaluate_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)

class TestPipeline(unittest.TestCase):
    def test_load_data(self):
        data = load_data('data/traffic.csv')
        self.assertIsNotNone(data)

    def test_preprocess_data(self):
        data = pd.DataFrame({
            'temp': [20, 22, 18],
            'rain_1h': [0, 1, 0],
            'snow_1h': [0, 0, 1],
            'clouds_all': [50, 70, 30],
            'weather_main': ['sunny', 'rainy', 'snowy'],
            'IsHoliday': [0, 1, 0],
            'Hour': [10, 14, 18],
            'DayOfWeek': [1, 3, 5],
            'Month': [1, 4, 7],
            'Congestion': ['low', 'medium', 'high']
        })
        X_train, X_test, y_train, y_test = preprocess_data(data)
        self.assertEqual(len(X_train), 2)
        self.assertEqual(len(y_train), 2)
        self.assertEqual(len(X_test), 1)
        self.assertEqual(len(y_test), 1)

    def test_train_model(self):
        data = pd.DataFrame({
            'temp': [20, 22, 18],
            'rain_1h': [0, 1, 0],
            'snow_1h': [0, 0, 1],
            'clouds_all': [50, 70, 30],
            'weather_main': ['sunny', 'rainy', 'snowy'],
            'IsHoliday': [0, 1, 0],
            'Hour': [10, 14, 18],
            'DayOfWeek': [1, 3, 5],
            'Month': [1, 4, 7],
            'Congestion': ['low', 'medium', 'high']
        })
        X_train, X_test, y_train, y_test = preprocess_data(data)
        model = train_model(X_train, y_train)
        self.assertIsInstance(model, RandomForestClassifier)

    def test_evaluate_model(self):
        data = pd.DataFrame({
            'temp': [20, 22, 18],
            'rain_1h': [0, 1, 0],
            'snow_1h': [0, 0, 1],
            'clouds_all': [50, 70, 30],
            'weather_main': ['sunny', 'rainy', 'snowy'],
            'IsHoliday': [0, 1, 0],
            'Hour': [10, 14, 18],
            'DayOfWeek': [1, 3, 5],
            'Month': [1, 4, 7],
            'Congestion': ['low', 'medium', 'high']
        })
        X_train, X_test, y_train, y_test = preprocess_data(data)
        model = train_model(X_train, y_train)
        accuracy = evaluate_model(model, X_test, y_test)
        self.assertGreater(accuracy, 0.5)

if __name__ == '__main__':
    unittest.main()
