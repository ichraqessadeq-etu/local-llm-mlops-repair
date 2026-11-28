from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import pytest
from pipeline import load_data, preprocess_data, train_model, evaluate_model, main

def test_load_data():
    """Test loading data."""
    with pytest.raises(ValueError):
        load_data("nonexistent_file.csv")

def test_preprocess_data():
    """Test preprocessing data."""
    data = pd.DataFrame({
        'PT08.S1(CO)': [1, 2, 3],
        'Anomaly': [0, 1, 0]
    })
    X_train, X_test, y_train, y_test = preprocess_data(data)
    assert X_train.shape == (2, 12)
    assert y_train.shape == (2,)
    assert X_test.shape == (1, 12)
    assert y_test.shape == (1,)

def test_train_model():
    """Test training the model."""
    data = pd.DataFrame({
        'PT08.S1(CO)': [1, 2, 3],
        'Anomaly': [0, 1, 0]
    })
    X_train, X_test, y_train, y_test = preprocess_data(data)
    model = train_model(X_train, y_train)
    assert isinstance(model, RandomForestClassifier)

def test_evaluate_model():
    """Test evaluating the model."""
    data = pd.DataFrame({
        'PT08.S1(CO)': [1, 2, 3],
        'Anomaly': [0, 1, 0]
    })
    X_train, X_test, y_train, y_test = preprocess_data(data)
    model = train_model(X_train, y_train)
    accuracy = evaluate_model(model, X_test, y_test)
    assert isinstance(accuracy, float)

def test_main():
    """Test the main function."""
    with pytest.raises(ValueError):
        main("nonexistent_file.csv")
