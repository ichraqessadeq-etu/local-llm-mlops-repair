import pytest
import pandas as pd
from pipeline import load_data, preprocess_data, train_model

def test_load_data():
    """Test the load_data function."""
    data = load_data('data/fraud.csv')
    assert isinstance(data, pd.DataFrame)

def test_preprocess_data():
    """Test the preprocess_data function."""
    data = pd.DataFrame({
        'Time': [1, 2, 3],
        'V1': [4, 5, 6],
        'Class': [0, 1, 0]
    })
    X, y = preprocess_data(data)
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)

def test_train_model():
    """Test the train_model function."""
    data = pd.DataFrame({
        'Time': [1, 2, 3],
        'V1': [4, 5, 6],
        'Class': [0, 1, 0]
    })
    X, y = preprocess_data(data)
    model = train_model(X, y)
    assert isinstance(model, RandomForestClassifier)
