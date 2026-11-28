import pytest
from pipeline import load_data, preprocess_data, train_model, run_pipeline

def test_load_data():
    file_path = 'data/fraud.csv'
    data = load_data(file_path)
    assert data is not None, "Data should not be None"

def test_preprocess_data():
    file_path = 'data/fraud.csv'
    data = load_data(file_path)
    X, y = preprocess_data(data)
    assert X is not None and y is not None, "X and y should not be None"

def test_train_model():
    file_path = 'data/fraud.csv'
    data = load_data(file_path)
    X, y = preprocess_data(data)
    model = train_model(X, y)
    assert model is not None, "Model should not be None"

def test_run_pipeline():
    file_path = 'data/fraud.csv'
    run_pipeline(file_path)
