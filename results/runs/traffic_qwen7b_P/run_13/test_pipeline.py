import pandas as pd
from pipeline import load_data, preprocess_data, train_model, evaluate_model

def test_load_data():
    data = load_data('data/traffic.csv')
    assert not data.empty
    assert 'Congestion' in data.columns

def test_preprocess_data():
    data = load_data('data/traffic.csv')
    X, y = preprocess_data(data)
    assert not X.empty
    assert not y.empty
    assert len(X.columns) == len(data.columns) - 1
    assert len(y) == len(data)

def test_train_model():
    data = load_data('data/traffic.csv')
    X, y = preprocess_data(data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = train_model(X_train, y_train)
    assert model is not None
    assert isinstance(model, RandomForestClassifier)

def test_evaluate_model():
    data = load_data('data/traffic.csv')
    X, y = preprocess_data(data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = train_model(X_train, y_train)
    accuracy = evaluate_model(model, X_test, y_test)
    assert accuracy is not None
    assert 0 <= accuracy <= 1
