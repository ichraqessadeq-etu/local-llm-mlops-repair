from sklearn.ensemble import RandomForestClassifier
import pytest
import pipeline
import pandas as pd

def test_load_data():
    """Test the load_data function."""
    file_path = 'data/traffic.csv'
    data = pipeline.load_data(file_path)
    assert isinstance(data, pd.DataFrame)

def test_preprocess_data():
    """Test the preprocess_data function."""
    file_path = 'data/traffic.csv'
    data = pipeline.load_data(file_path)
    X, y = pipeline.preprocess_data(data)
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert X.shape[1] == 10
    assert y.shape[0] == len(data)

def test_train_model():
    """Test the train_model function."""
    file_path = 'data/traffic.csv'
    data = pipeline.load_data(file_path)
    X, y = pipeline.preprocess_data(data)
    model = pipeline.train_model(X, y)
    assert isinstance(model, RandomForestClassifier)
    assert model.n_estimators > 0
