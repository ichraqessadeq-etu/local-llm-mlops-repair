import pytest
from pipeline import load_data, preprocess_data, train_model

def test_load_data():
    assert load_data("data/fraud.csv").shape[0] > 0

def test_preprocess_data():
    # Add your test cases for preprocess_data

def test_train_model():
    # Add your test cases for train_model
    # You can use mock data or mock mlflow to test this function
