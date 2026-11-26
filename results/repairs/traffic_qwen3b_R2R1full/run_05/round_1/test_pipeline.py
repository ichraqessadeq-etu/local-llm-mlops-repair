# test_pipeline.py
from sklearn.ensemble import RandomForestClassifier
import pytest
import pandas as pd
from pipeline import load_data, preprocess_data, train_model, evaluate_model

@pytest.fixture
def data():
    """Fixture to load and preprocess data."""
    file_path = 'data/traffic.csv'
    data = load_data(file_path)
    X, y = preprocess_data(data)
    return X, y

def test_load_data():
    """Test loading data."""
    with pytest.raises(ValueError):
        load_data('non_existent_file.csv')
    with pytest.raises(ValueError):
        load_data('empty_file.csv')

def test_preprocess_data():
    """Test preprocessing data."""
    with pytest.raises(ValueError):
        preprocess_data(pd.DataFrame())

def test_train_model():
    """Test training the model."""
    X, y = data()
    model = train_model(X, y)
    assert isinstance(model, RandomForestClassifier)

def test_evaluate_model():
    """Test evaluating the model."""
    X, y = data()
    model = train_model(X, y)
    accuracy = evaluate_model(model, X, y)
    assert isinstance(accuracy, float)
