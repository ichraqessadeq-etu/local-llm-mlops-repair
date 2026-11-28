from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import pytest
from pipeline import load_data, preprocess_data, train_model, evaluate_model

@pytest.fixture
def data():
    # Load a sample dataset
    data = pd.read_csv('data/air_quality.csv')
    return data

def test_load_data(data):
    """Test loading data."""
    assert isinstance(load_data('data/air_quality.csv'), pd.DataFrame)

def test_preprocess_data(data):
    """Test preprocessing data."""
    X_train, X_test, y_train, y_test = preprocess_data(data)
    assert isinstance(X_train, pd.DataFrame)
    assert isinstance(X_test, pd.DataFrame)
    assert isinstance(y_train, pd.Series)
    assert isinstance(y_test, pd.Series)

def test_train_model(data):
    """Test training the model."""
    X_train, X_test, y_train, y_test = preprocess_data(data)
    rf = train_model(X_train, y_train)
    assert isinstance(rf, RandomForestClassifier)

def test_evaluate_model(data):
    """Test evaluating the model."""
    X_train, X_test, y_train, y_test = preprocess_data(data)
    rf = train_model(X_train, y_train)
    accuracy = evaluate_model(rf, X_test, y_test)
    assert isinstance(accuracy, float)
