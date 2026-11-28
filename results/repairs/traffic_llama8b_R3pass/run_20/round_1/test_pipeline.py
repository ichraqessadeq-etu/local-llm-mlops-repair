from sklearn.ensemble import RandomForestClassifier
import pytest
from pipeline import load_data, preprocess_data, train_model

def test_load_data():
    data = load_data('data/traffic.csv')
    assert data.shape[0] > 0

def test_preprocess_data():
    data = load_data('data/traffic.csv')
    X, y = preprocess_data(data)
    assert X.shape[0] > 0
    assert y.shape[0] > 0

def test_train_model():
    data = load_data('data/traffic.csv')
    X, y = preprocess_data(data)
    model = train_model(X, y)
    assert isinstance(model, RandomForestClassifier)
