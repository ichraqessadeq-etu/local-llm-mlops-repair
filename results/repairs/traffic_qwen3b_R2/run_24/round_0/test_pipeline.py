import pytest
from pipeline import load_data, preprocess_data, train_model, evaluate_model, run_pipeline

def test_load_data():
    """Test loading data."""
    data = load_data('data/traffic.csv')
    assert isinstance(data, pd.DataFrame)

def test_preprocess_data():
    """Test preprocessing data."""
    data = load_data('data/traffic.csv')
    X_train, y_train, X_test, y_test = preprocess_data(data)
    assert isinstance(X_train, pd.DataFrame)
    assert isinstance(y_train, pd.Series)
    assert isinstance(X_test, pd.DataFrame)
    assert isinstance(y_test, pd.Series)

def test_train_model():
    """Test training the model."""
    data = load_data('data/traffic.csv')
    X_train, y_train, X_test, y_test = preprocess_data(data)
    model = train_model(X_train, y_train)
    assert isinstance(model, RandomForestClassifier)

def test_evaluate_model():
    """Test evaluating the model."""
    data = load_data('data/traffic.csv')
    X_train, y_train, X_test, y_test = preprocess_data(data)
    model = train_model(X_train, y_train)
    accuracy = evaluate_model(model, X_test, y_test)
    assert isinstance(accuracy, float)

def test_run_pipeline():
    """Test running the entire pipeline."""
    run_pipeline('data/traffic.csv')
