from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import pytest
from pipeline import load_data, preprocess_data, train_model, evaluate_model, run_pipeline

@pytest.fixture
def test_data():
    # Create a sample dataset for testing
    data = {
        'Time': [1, 2, 3, 4, 5],
        'V1': [10, 20, 30, 40, 50],
        'V2': [100, 200, 300, 400, 500],
        'Class': [0, 1, 0, 1, 0]
    }
    return pd.DataFrame(data)

def test_load_data(test_data):
    assert load_data(test_data) is not None

def test_preprocess_data(test_data):
    X, y = preprocess_data(test_data)
    assert X.shape == (5, 2)
    assert y.shape == (5,)

def test_train_model(test_data):
    X, y = preprocess_data(test_data)
    model = train_model(X, y)
    assert isinstance(model, RandomForestClassifier)

def test_evaluate_model(test_data):
    X, y = preprocess_data(test_data)
    model = train_model(X, y)
    accuracy = evaluate_model(model, X, y)
    assert 0 <= accuracy <= 1

def test_run_pipeline(test_data):
    run_pipeline(test_data)
