import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import unittest
from pipeline import load_data, preprocess_data, train_model
import shutil
import os

Features = ['PT08.S1(CO)', 'PT08.S2(NMHC)', 'PT08.S3(NOx)', 'PT08.S4(NO2)', 'PT08.S5(O3)', 'NOx(GT)', 'C6H6(GT)', 'T', 'RH', 'AH', 'Hour', 'DayOfWeek', 'Month']

class TestPipeline(unittest.TestCase):
    def setUp(self):
        # Mock model file for tests
        model_path = os.path.join('models', f'random_forest_mock.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(RandomForestClassifier(), f)

    def test_load_data(self):
        data = load_data("data/air_quality.csv")
        self.assertIsInstance(data, pd.DataFrame)

    def test_preprocess_data(self):
        data = load_data("data/air_quality.csv")
        X, y = preprocess_data(data)
        self.assertIsInstance(X, pd.DataFrame)
        self.assertIsInstance(y, pd.Series)

    def test_train_model(self):
        data = load_data("data/air_quality.csv")
        X, y = preprocess_data(data)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        model = train_model(X_train, y_train)
        # Add your custom assertions to check the model's performance
