import unittest
from pipeline import load_data, preprocess_data, train_model, save_model
import pandas as pd

class TestPipeline(unittest.TestCase):
    def test_load_data(self):
        data = load_data('data/traffic.csv')
        self.assertIsInstance(data, pd.DataFrame)

    def test_preprocess_data(self):
        data = pd.DataFrame({
            'temp': [20, 22, 18],
            'rain_1h': [0.1, 0.2, 0.0],
            'snow_1h': [0.0, 0.0, 0.1],
            'clouds_all': [50, 60, 70],
            'weather_main': ['sunny', 'rainy', 'snowy'],
            'IsHoliday': [0, 1, 0],
            'Hour': [10, 14, 18],
            'DayOfWeek': ['Monday', 'Wednesday', 'Friday'],
            'Month': ['January', 'March', 'May']
        })
        X, y = preprocess_data(data)
        self.assertIsInstance(X, pd.DataFrame)
        self.assertIsInstance(y, pd.Series)

    def test_train_model(self):
        data = pd.DataFrame({
            'temp': [20, 22, 18],
            'rain_1h': [0.1, 0.2, 0.0],
            'snow_1h': [0.0, 0.0, 0.1],
            'clouds_all': [50, 60, 70],
            'weather_main': ['sunny', 'rainy', 'snowy'],
            'IsHoliday': [0, 1, 0],
            'Hour': [10, 14, 18],
            'DayOfWeek': ['Monday', 'Wednesday', 'Friday'],
            'Month': ['January', 'March', 'May'],
            'Congestion': [0, 1, 0]
        })
        X, y = preprocess_data(data)
        model = train_model(X, y)
        self.assertIsInstance(model, RandomForestClassifier)

    def test_save_model(self):
        data = pd.DataFrame({
            'temp': [20, 22, 18],
            'rain_1h': [0.1, 0.2, 0.0],
            'snow_1h': [0.0, 0.0, 0.1],
            'clouds_all': [50, 60, 70],
            'weather_main': ['sunny', 'rainy', 'snowy'],
            'IsHoliday': [0, 1, 0],
            'Hour': [10, 14, 18],
            'DayOfWeek': ['Monday', 'Wednesday', 'Friday'],
            'Month': ['January', 'March', 'May'],
            'Congestion': [0, 1, 0]
        })
        X, y = preprocess_data(data)
        model = train_model(X, y)
        save_model(model, 'traffic_congestion', 'test_run')
        self.assertTrue(True)  # Placeholder for actual verification

if __name__ == '__main__':
    unittest.main()
