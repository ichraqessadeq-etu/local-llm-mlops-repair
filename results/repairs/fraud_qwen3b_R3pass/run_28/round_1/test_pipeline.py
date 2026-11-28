from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import pytest
from pipeline import load_data, preprocess_data, train_model, evaluate_model

def test_load_data():
    """Test loading the dataset."""
    with pytest.raises(ValueError):
        load_data('nonexistent_file.csv')

def test_preprocess_data():
    """Test preprocessing the data."""
    data = pd.DataFrame({
        'Class': [0, 1, 0, 1],
        'V1': [1, 2, 3, 4]
    })
    X, y = preprocess_data(data)
    assert X.shape == (4, 1)
    assert y.shape == (4,)

def test_train_model():
    """Test training the model."""
    data = pd.DataFrame({
        'Class': [0, 1, 0, 1],
        'V1': [1, 2, 3, 4]
    })
    X, y = preprocess_data(data)
    model = train_model(X, y)
    assert isinstance(model, RandomForestClassifier)

def test_evaluate_model():
    """Test evaluating the model."""
    data = pd.DataFrame({
        'Class': [0, 1, 0, 1],
        'V1': [1, 2, 3, 4]
    })
    X, y = preprocess_data(data)
    model = train_model(X, y)
    accuracy = evaluate_model(model, X, y)
    assert isinstance(accuracy, float)
