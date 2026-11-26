import unittest
from pipeline import load_data, preprocess_data, train_model

class TestPipeline(unittest.TestCase):
    def test_load_data(self):
        # Test loading data from the correct file
        data = load_data("data/air_quality.csv")
        # Add assertions to check if the data is loaded correctly

    def test_preprocess_data(self):
        # Test preprocessing data with mock data
        # Add assertions to check if the preprocessing works as expected

    def test_train_model(self):
        # Test training the model with mock data
        # Add assertions to check if the model is trained correctly
        # You might need to mock the MLflow API for this test

        X, y = preprocess_data(load_data("data/air_quality.csv"))
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        train_model(X_train, y_train)
