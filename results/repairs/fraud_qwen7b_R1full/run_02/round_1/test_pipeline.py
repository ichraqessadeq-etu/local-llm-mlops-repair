import pandas as pd
from sklearn.model_selection import train_test_split
from pipeline import load_data, preprocess_data, train_model

def test_load_data():
    """Test data loading function."""
    data = load_data('data/fraud.csv')
    assert not data.empty, "Data is empty"
    assert 'Class' in data.columns, "Target column 'Class' is missing"

def test_preprocess_data():
    """Test data preprocessing function."""
    data = load_data('data/fraud.csv')
    X, y = preprocess_data(data)
    assert not X.empty, "Features are empty"
    assert not y.empty, "Target is empty"
    assert len(X) == len(y), "Feature and target lengths do not match"

def test_train_model():
    """Test model training function."""
    data = load_data('data/fraud.csv')
    X, y = preprocess_data(data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = train_model(X_train, y_train)
    assert model is not None, "Model is None"
    assert hasattr(model, 'predict'), "Model does not have a predict method"
