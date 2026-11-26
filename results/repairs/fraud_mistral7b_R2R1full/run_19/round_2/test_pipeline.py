from sklearn.model_selection import train_test_split
import unittest
from pipeline import load_data, preprocess_data, train_model

class TestPipeline(unittest.TestCase):
    def test_load_data(self):
        data = load_data('data/fraud.csv')
        # Add assertions to check if the data is loaded correctly

    def test_preprocess_data(self):
        data = load_data('data/fraud.csv')
        X, y = preprocess_data(data)
        # Add assertions to check if the preprocessing is done correctly

    def test_train_model(self):
        data = load_data('data/fraud.csv')
        X, y = preprocess_data(data)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        model = train_model(X_train, y_train)
        # Add assertions to check if the model is trained correctly and saves the model artifact
