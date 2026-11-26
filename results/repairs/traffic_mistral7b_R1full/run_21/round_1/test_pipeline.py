import unittest
import tempfile
import pandas as pd
from pipeline import load_data, preprocess_data, train_model

class TestPipeline(unittest.TestCase):
    def setUp(self):
        self.data = pd.DataFrame({
            'temp': [20, 25, 30, 22, 28],
            'rain_1h': [0, 1, 0, 1, 0],
            'snow_1h': [0, 0, 0, 0, 0],
            'clouds_all': [75, 50, 25, 100, 75],
            'weather_main': ['Clear', 'Rain', 'Clear', 'Rain', 'Clear'],
            'IsHoliday': [0, 0, 0, 1, 0],
            'Hour': [1, 2, 3, 4, 5],
            'DayOfWeek': [1, 2, 3, 4, 5],
            'Month': ['Jan', 'Jan', 'Jan', 'Jan', 'Jan'],
            'Congestion': [0, 1, 0, 1, 0]
        })

    def tearDown(self):
        pass

    def test_load_data(self):
        data = load_data("data/traffic.csv")
        self.assertIsInstance(data, pd.DataFrame)

    def test_preprocess_data(self):
        X, y = preprocess_data(self.data)
        self.assertIsInstance(X, pd.DataFrame)
        self.assertIsInstance(y, pd.Series)

    def test_train_model(self):
        X_train, y_train = preprocess_data(self.data)
        model = train_model(X_train, y_train)
        self.assertIsInstance(model, RandomForestClassifier)

if __name__ == "__main__":
    unittest.main()
