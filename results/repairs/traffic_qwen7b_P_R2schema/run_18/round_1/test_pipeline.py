from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
from pipeline import load_data, preprocess_data, train_model

def test_load_data():
    """Test the load_data function."""
    data = load_data('data/traffic.csv')
    assert not data.empty
    assert 'Congestion' in data.columns

def test_preprocess_data():
    """Test the preprocess_data function."""
    data = load_data('data/traffic.csv')
    X, y = preprocess_data(data)
    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert X.shape[0] == y.shape[0]

def test_train_model():
    """Test the train_model function."""
    data = load_data('data/traffic.csv')
    X, y = preprocess_data(data)
    model = train_model(X, y)
    assert isinstance(model, RandomForestClassifier)
    assert model.n_estimators == 100
    assert model.random_state == 42
