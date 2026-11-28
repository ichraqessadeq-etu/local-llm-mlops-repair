from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import pytest
from pipeline import load_data, preprocess_data, train_model, evaluate_model

@pytest.fixture
def data():
    """Fixture to load and preprocess data."""
    file_path = 'data/traffic.csv'
    data = load_data(file_path)
    X_train, X_test, y_train, y_test = preprocess_data(data)
    return X_train, X_test, y_train, y_test

def test_load_data():
    """Test loading data."""
    with pytest.raises(FileNotFoundError):
        load_data('nonexistent_file.csv')
    with pytest.raises(pd.errors.EmptyDataError):
        load_data('data/empty.csv')

def test_preprocess_data():
    """Test preprocessing data."""
    X_train, X_test, y_train, y_test = data()
    assert isinstance(X_train, pd.DataFrame)
    assert isinstance(X_test, pd.DataFrame)
    assert isinstance(y_train, pd.Series)
    assert isinstance(y_test, pd.Series)

def test_train_model():
    """Test training the model."""
    X_train, X_test, y_train, y_test = data()
    model = train_model(X_train, y_train)
    assert isinstance(model, RandomForestClassifier)

def test_evaluate_model():
    """Test evaluating the model."""
    X_train, X_test, y_train, y_test = data()
    model = train_model(X_train, y_train)
    accuracy = evaluate_model(model, X_test, y_test)
    assert isinstance(accuracy, float)
