# test_pipeline.py
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import pytest
from pipeline import load_data, preprocess_data, train_model, save_model

def test_load_data():
    """Test loading the dataset."""
    data = load_data('data/fraud.csv')
    assert isinstance(data, pd.DataFrame)

def test_preprocess_data():
    """Test preprocessing the dataset."""
    data = load_data('data/fraud.csv')
    X_train, X_test, y_train, y_test = preprocess_data(data)
    assert isinstance(X_train, pd.DataFrame)
    assert isinstance(X_test, pd.DataFrame)
    assert isinstance(y_train, pd.Series)
    assert isinstance(y_test, pd.Series)

def test_train_model():
    """Test training the model."""
    data = load_data('data/fraud.csv')
    X_train, X_test, y_train, y_test = preprocess_data(data)
    model = train_model(X_train, y_train, X_test)
    assert isinstance(model, RandomForestClassifier)

def test_save_model():
    """Test saving the model."""
    data = load_data('data/fraud.csv')
    X_train, X_test, y_train, y_test = preprocess_data(data)
    model = train_model(X_train, y_train, X_test)
    save_model(model, 'model')
