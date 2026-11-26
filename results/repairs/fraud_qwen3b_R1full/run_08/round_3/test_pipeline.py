import pytest
import pandas as pd
from pipeline import load_data, preprocess_data, train_model

def test_load_data():
    """Test loading data."""
    data = load_data('data/fraud.csv')
    assert isinstance(data, pd.DataFrame), "Data should be a pandas DataFrame."

def test_preprocess_data():
    """Test preprocessing data."""
    data = load_data('data/fraud.csv')
    X_train, y_train, X_test, y_test = preprocess_data(data)
    assert isinstance(X_train, pd.DataFrame), "X_train should be a pandas DataFrame."
    assert isinstance(y_train, pd.Series), "y_train should be a pandas Series."
    assert isinstance(X_test, pd.DataFrame), "X_test should be a pandas DataFrame."
    assert isinstance(y_test, pd.Series), "y_test should be a pandas Series."

def test_train_model():
    """Test training the model."""
    data = load_data('data/fraud.csv')
    X_train, y_train, _, _ = preprocess_data(data)
    model = train_model(X_train, y_train)
    assert isinstance(model, RandomForestClassifier), "Model should be a RandomForestClassifier."
