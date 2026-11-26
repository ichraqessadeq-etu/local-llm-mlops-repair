import pytest
from pipeline import load_data, preprocess_data, train_model

def test_load_data():
    """Test loading data."""
    data = load_data('data/traffic.csv')
    assert isinstance(data, pd.DataFrame)

def test_preprocess_data():
    """Test preprocessing data."""
    data = load_data('data/traffic.csv')
    X, y = preprocess_data(data)
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)

def test_train_model():
    """Test training the model."""
    data = load_data('data/traffic.csv')
    X, y = preprocess_data(data)
    model = train_model(X, y)
    assert isinstance(model, RandomForestClassifier)
