import unittest
from pipeline import load_data, preprocess_data, train_model
import pandas as pd

class TestPipeline(unittest.TestCase):
    
    def test_load_data(self):
        data = load_data('data/traffic.csv')
        self.assertIsNotNone(data)
    
    def test_preprocess_data(self):
        data = pd.DataFrame({
            'temp': [20, 22, 19],
            'rain_1h': [0.5, 0.3, 0.1],
            'snow_1h': [0, 0, 0],
            'clouds_all': [50, 70, 30],
            'weather_main': ['sunny', 'cloudy', 'rainy'],
            'IsHoliday': [False, True, False],
            'Hour': [12, 15, 18],
            'DayOfWeek': ['Monday', 'Wednesday', 'Friday'],
            'Month': [1, 3, 5],
            'Congestion': ['high', 'low', 'high']
        })
        X_train, X_test, y_train, y_test = preprocess_data(data)
        self.assertEqual(len(X_train), 2)
        self.assertEqual(len(y_train), 2)
    
    def test_train_model(self):
        data = pd.DataFrame({
            'temp': [20, 22, 19],
            'rain_1h': [0.5, 0.3, 0.1],
            'snow_1h': [0, 0, 0],
            'clouds_all': [50, 70, 30],
            'weather_main': ['sunny', 'cloudy', 'rainy'],
            'IsHoliday': [False, True, False],
            'Hour': [12, 15, 18],
            'DayOfWeek': ['Monday', 'Wednesday', 'Friday'],
            'Month': [1, 3, 5],
            'Congestion': ['high', 'low', 'high']
        })
        X_train, X_test, y_train, y_test = preprocess_data(data)
        model = train_model(X_train, y_test)  # Corrected to use y_test instead of y_train
        self.assertIsNotNone(model)

if __name__ == '__main__':
    unittest.main()
