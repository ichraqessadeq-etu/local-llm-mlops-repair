from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import pytest
from pipeline import load_data, preprocess_data, train_model, evaluate_model, run_pipeline

@pytest.fixture
def data():
    """Fixture to load and preprocess the data."""
    file_path = 'data/fraud.csv'
    data = load_data(file_path)
    X_train, X_test, y_train, y_test = preprocess_data(data)
    return X_train, X_test, y_train, y_test

def test_load_data():
    """Test the load_data function."""
    file_path = 'data/fraud.csv'
    data = load_data(file_path)
    assert isinstance(data, pd.DataFrame)

def test_preprocess_data(data):
    """Test the preprocess_data function."""
    X_train, X_test, y_train, y_test = data
    assert isinstance(X_train, pd.DataFrame)
    assert isinstance(X_test, pd.DataFrame)
    assert isinstance(y_train, pd.Series)
    assert isinstance(y_test, pd.Series)

def test_train_model(data):
    """Test the train_model function."""
    X_train, X_test, y_train, y_test = data
    model = train_model(X_train, y_train)
    assert isinstance(model, RandomForestClassifier)

def test_evaluate_model(data):
    """Test the evaluate_model function."""
    X_train, X_test, y_train, y_test = data
    model = train_model(X_train, y_train)
    accuracy = evaluate_model(model, X_test, y_test)
    assert isinstance(accuracy, float)

def test_run_pipeline(data):
    """Test the run_pipeline function."""
    X_train, X_test, y_train, y_test = data
    run_pipeline('data/fraud.csv')
