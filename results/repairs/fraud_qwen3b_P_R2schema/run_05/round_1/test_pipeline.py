from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import pytest
from pipeline import load_data, preprocess_data, train_model, save_model

def test_load_data():
    """Test loading data."""
    data = load_data("data/fraud.csv")
    assert isinstance(data, pd.DataFrame)

def test_preprocess_data():
    """Test preprocessing data."""
    data = load_data("data/fraud.csv")
    X, y = preprocess_data(data)
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)

def test_train_model():
    """Test training the model."""
    data = load_data("data/fraud.csv")
    X, y = preprocess_data(data)
    model = train_model(X, y)
    assert isinstance(model, RandomForestClassifier)

def test_save_model():
    """Test saving the model."""
    data = load_data("data/fraud.csv")
    X, y = preprocess_data(data)
    model = train_model(X, y)
    experiment_name = "fraud_detection"
    run_name = "run_1"
    run_id = save_model(model, experiment_name, run_name)
    assert isinstance(run_id, str)
