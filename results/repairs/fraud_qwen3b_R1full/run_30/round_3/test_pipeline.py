import pytest
import pandas as pd
from pipeline import load_data, preprocess_data, train_model, evaluate_model

def test_load_data():
    """Test loading data from a CSV file."""
    data = load_data('data/fraud.csv')
    assert isinstance(data, pd.DataFrame)

def test_preprocess_data():
    """Test preprocessing the data."""
    data = pd.DataFrame({
        'Time': [1, 2, 3],
        'V1': [4, 5, 6],
        'Class': [0, 1, 0]
    })
    X, y = preprocess_data(data)
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert len(X) == 3
    assert len(y) == 3

def test_train_model():
    """Test training the model."""
    data = pd.DataFrame({
        'Time': [1, 2, 3],
        'V1': [4, 5, 6],
        'Class': [0, 1, 0]
    })
    X, y = preprocess_data(data)
    model = train_model(X, y)
    assert isinstance(model, RandomForestClassifier)

def test_evaluate_model():
    """Test evaluating the model."""
    data = pd.DataFrame({
        'Time': [1, 2, 3],
        'V1': [4, 5, 6],
        'Class': [0, 1, 0]
    })
    X, y = preprocess_data(data)
    model = train_model(X, y)
    accuracy = evaluate_model(model, X, y)
    assert isinstance(accuracy, float)
