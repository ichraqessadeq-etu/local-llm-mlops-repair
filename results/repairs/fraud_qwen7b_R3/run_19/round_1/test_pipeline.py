from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from pipeline import load_data, preprocess_data, train_model

def test_load_data():
    """Test the load_data function."""
    data = load_data('data/fraud.csv')
    assert not data.empty
    assert 'Class' in data.columns

def test_preprocess_data():
    """Test the preprocess_data function."""
    data = load_data('data/fraud.csv')
    X, y = preprocess_data(data)
    assert X.shape[0] == y.shape[0]
    assert 'Class' not in X.columns

def test_train_model():
    """Test the train_model function."""
    data = load_data('data/fraud.csv')
    X, y = preprocess_data(data)
    model = train_model(X, y)
    assert model is not None
    assert isinstance(model, RandomForestClassifier)
