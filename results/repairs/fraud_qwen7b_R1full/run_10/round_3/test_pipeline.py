# test_pipeline.py

import pandas as pd
from pipeline import load_data, preprocess_data, train_model
from sklearn.ensemble import RandomForestClassifier  # Import RandomForestClassifier
import mlflow  # Import mlflow

def test_load_data():
    """Test loading data."""
    data = load_data('data/fraud.csv')
    assert isinstance(data, pd.DataFrame)
    assert 'Class' in data.columns

def test_preprocess_data():
    """Test preprocessing data."""
    data = load_data('data/fraud.csv')
    X, y = preprocess_data(data)
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert len(X) == len(y)

def test_train_model():
    """Test training model."""
    data = load_data('data/fraud.csv')
    X, y = preprocess_data(data)
    model = train_model(X, y)
    assert isinstance(model, RandomForestClassifier)
    assert 'model' in mlflow.active_run().artifacts
