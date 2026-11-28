from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import pytest
from pipeline import load_data, preprocess_data, train_model, evaluate_model, run_pipeline

@pytest.fixture
def test_data():
    """Fixture to load test data."""
    file_path = 'data/air_quality.csv'
    return load_data(file_path)

def test_load_data(test_data):
    """Test the load_data function."""
    assert isinstance(test_data, pd.DataFrame)

def test_preprocess_data(test_data):
    """Test the preprocess_data function."""
    X, y = preprocess_data(test_data)
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)

def test_train_model():
    """Test the train_model function."""
    X, y = (pd.DataFrame(), pd.Series())
    model = train_model(X, y)
    assert isinstance(model, RandomForestClassifier)

def test_evaluate_model():
    """Test the evaluate_model function."""
    model = RandomForestClassifier(random_state=42)
    X, y = (pd.DataFrame(), pd.Series())
    accuracy = evaluate_model(model, X, y)
    assert isinstance(accuracy, float)

def test_run_pipeline(test_data):
    """Test the run_pipeline function."""
    model = run_pipeline('data/air_quality.csv')
    assert model is not None