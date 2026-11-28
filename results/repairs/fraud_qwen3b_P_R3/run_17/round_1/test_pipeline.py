from sklearn.ensemble import RandomForestClassifier
import pytest
from pipeline import load_data, preprocess_data, train_model
import pandas as pd

@pytest.fixture
def test_data():
    data = pd.DataFrame({
        'Time': [1, 2, 3],
        'V1': [4, 5, 6],
        'V2': [7, 8, 9],
        'Class': [0, 1, 0]
    })
    return data

def test_load_data(test_data):
    """Test loading data from a CSV file."""
    file_path = 'data/fraud.csv'
    data = load_data(file_path)
    assert data.equals(test_data)

def test_preprocess_data(test_data):
    """Test preprocessing the data."""
    X, y = preprocess_data(test_data)
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert X.shape == (3, 2)
    assert y.shape == (3,)

def test_train_model(test_data):
    """Test training the model."""
    X, y = preprocess_data(test_data)
    model = train_model(X, y)
    assert isinstance(model, RandomForestClassifier)

def test_main():
    """Test the main function."""
    test_data = pd.DataFrame({
        'Time': [1, 2, 3],
        'V1': [4, 5, 6],
        'V2': [7, 8, 9],
        'Class': [0, 1, 0]
    })
    with pytest.raises(ValueError):
        load_data('non_existent_file.csv')
    with pytest.raises(ValueError):
        load_data('data/empty.csv')
