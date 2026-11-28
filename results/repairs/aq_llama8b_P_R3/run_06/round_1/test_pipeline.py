from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
import unittest
from pipeline import load_data, preprocess_data, train_model

class TestPipeline(unittest.TestCase):
    def test_load_data(self):
        data = load_data('data/air_quality.csv')
        self.assertIsInstance(data, pd.DataFrame)
    
    def test_preprocess_data(self):
        data = load_data('data/air_quality.csv')
        X_train, X_test, y_train, y_test = preprocess_data(data)
        self.assertIsInstance(X_train, np.ndarray)
        self.assertIsInstance(X_test, np.ndarray)
        self.assertIsInstance(y_train, np.ndarray)
        self.assertIsInstance(y_test, np.ndarray)
    
    def test_train_model(self):
        data = load_data('data/air_quality.csv')
        X_train, X_test, y_train, y_test = preprocess_data(data)
        model = train_model(X_train, y_train)
        self.assertIsInstance(model, RandomForestClassifier)
