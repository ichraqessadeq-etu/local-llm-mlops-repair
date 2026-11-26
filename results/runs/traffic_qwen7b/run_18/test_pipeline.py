import pandas as pd
from pipeline import load_data, preprocess_data, train_model

def test_load_data():
    """Test data loading."""
    data = load_data('data/traffic.csv')
    assert not data.empty
    assert 'Congestion' in data.columns

def test_preprocess_data():
    """Test data preprocessing."""
    data = load_data('data/traffic.csv')
    X_train, X_test, y_train, y_test = preprocess_data(data)
    assert not X_train.empty
    assert not y_train.empty
    assert not X_test.empty
    assert not y_test.empty

def test_train_model():
    """Test model training."""
    data = load_data('data/traffic.csv')
    X_train, X_test, y_train, y_test = preprocess_data(data)
    model = train_model(X_train, y_train)
    assert model is not None
    assert hasattr(model, 'predict')
