import pytest
from pipeline import load_data, preprocess_data, train_model

def test_load_data():
    data = load_data('data/fraud.csv')
    assert data.shape[0] > 0

def test_preprocess_data():
    data = load_data('data/fraud.csv')
    X_train, X_test, y_train, y_test = preprocess_data(data)
    assert X_train.shape[0] > 0
    assert y_train.shape[0] > 0

def test_train_model():
    data = load_data('data/fraud.csv')
    X_train, X_test, y_train, y_test = preprocess_data(data)
    model = train_model(X_train, y_train)
    assert isinstance(model, RandomForestClassifier)
