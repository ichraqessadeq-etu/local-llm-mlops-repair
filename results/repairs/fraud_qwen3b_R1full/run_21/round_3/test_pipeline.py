import pytest
import pandas as pd
from pipeline import load_data, preprocess_data, train_model, evaluate_model
from sklearn.ensemble import RandomForestClassifier

def test_load_data():
    """Test loading the dataset."""
    with pytest.raises(ValueError):
        load_data('non_existent_file.csv')

    with pytest.raises(ValueError):
        load_data('empty.csv')

    with pytest.raises(ValueError):
        load_data('invalid.csv')

def test_preprocess_data():
    """Test preprocessing the data."""
    data = pd.DataFrame({
        'Time': [1, 2, 3],
        'V1': [4, 5, 6],
        'Class': [0, 1, 0]
    })
    X, y = preprocess_data(data)
    assert X.shape == (3, 2)
    assert X.columns.tolist() == ['Time', 'V1']
    assert y.shape == (3,)

def test_train_model():
    """Test training the model."""
    X = pd.DataFrame({
        'V1': [1, 2, 3]
    })
    y = pd.Series([0, 1, 0])
    model = train_model(X, y)
    assert isinstance(model, RandomForestClassifier)

def test_evaluate_model():
    """Test evaluating the model."""
    X = pd.DataFrame({
        'V1': [1, 2, 3]
    })
    y = pd.Series([0, 1, 0])
    model = train_model(X, y)
    accuracy = evaluate_model(model, X, y)
    assert accuracy == 1.0
