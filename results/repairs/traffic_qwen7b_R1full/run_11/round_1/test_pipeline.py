import pandas as pd
from sklearn.model_selection import train_test_split
from pipeline import load_data, preprocess_data, train_model

def test_load_data():
    """Test the load_data function."""
    data = load_data('data/traffic.csv')
    assert not data.empty, "Data should not be empty"

def test_preprocess_data():
    """Test the preprocess_data function."""
    data = load_data('data/traffic.csv')
    X, y = preprocess_data(data)
    assert not X.empty, "Features should not be empty"
    assert not y.empty, "Target should not be empty"

def test_train_model():
    """Test the train_model function."""
    data = load_data('data/traffic.csv')
    X, y = preprocess_data(data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = train_model(X_train, y_train)
    assert model is not None, "Model should not be None"
