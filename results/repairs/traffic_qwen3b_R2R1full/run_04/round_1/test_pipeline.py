from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import pytest
from pipeline import load_data, preprocess_data, train_model, evaluate_model

def test_load_data():
    """Test the load_data function."""
    data = load_data('data/traffic.csv')
    assert isinstance(data, pd.DataFrame)
    assert 'Congestion' in data.columns

def test_preprocess_data():
    """Test the preprocess_data function."""
    data = load_data('data/traffic.csv')
    X_train, X_test, y_train, y_test = preprocess_data(data)
    assert isinstance(X_train, pd.DataFrame)
    assert isinstance(X_test, pd.DataFrame)
    assert isinstance(y_train, pd.Series)
    assert isinstance(y_test, pd.Series)

def test_train_model():
    """Test the train_model function."""
    data = load_data('data/traffic.csv')
    X_train, X_test, y_train, y_test = preprocess_data(data)
    model = train_model(X_train, y_test)  # Changed y_train to y_test
    assert isinstance(model, RandomForestClassifier)

def test_evaluate_model():
    """Test the evaluate_model function."""
    data = load_data('data/traffic.csv')
    X_train, X_test, y_train, y_test = preprocess_data(data)
    model = train_model(X_train, y_test)  # Changed y_train to y_test
    evaluate_model(model, X_test, y_test)
