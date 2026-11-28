from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import pytest
from pipeline import load_data, preprocess_data, train_model, evaluate_model, run_pipeline

@pytest.fixture
def data():
    """Fixture to load and preprocess data."""
    file_path = 'data/traffic.csv'
    data = load_data(file_path)
    X, y = preprocess_data(data)
    return X, y

def test_load_data():
    """Test data loading."""
    file_path = 'data/traffic.csv'
    data = load_data(file_path)
    assert isinstance(data, pd.DataFrame)

def test_preprocess_data(data):
    """Test data preprocessing."""
    X, y = data
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert X.shape[0] == y.shape[0]

def test_train_model(data):
    """Test model training."""
    X, y = data
    model = train_model(X, y)
    assert isinstance(model, RandomForestClassifier)

def test_evaluate_model(data):
    """Test model evaluation."""
    X, y = data
    model = train_model(X, y)
    evaluate_model(model, X, y)

def test_run_pipeline(data):
    """Test the entire pipeline."""
    X, y = data
    model = run_pipeline('data/traffic.csv')
    assert isinstance(model, RandomForestClassifier)
