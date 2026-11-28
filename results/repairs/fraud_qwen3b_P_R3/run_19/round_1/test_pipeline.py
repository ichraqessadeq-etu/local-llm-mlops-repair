import pandas as pd
import pytest
from pipeline import load_data, preprocess_data, train_model, save_model
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow

@pytest.fixture
def data():
    """Fixture to load and preprocess data."""
    file_path = 'data/fraud.csv'
    data = load_data(file_path)
    X, y = preprocess_data(data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def test_load_data():
    """Test loading data."""
    file_path = 'data/fraud.csv'
    data = load_data(file_path)
    assert isinstance(data, pd.DataFrame)

def test_preprocess_data():
    """Test preprocessing data."""
    file_path = 'data/fraud.csv'
    data = load_data(file_path)
    X, y = preprocess_data(data)
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)

def test_train_model():
    """Test training the model."""
    file_path = 'data/fraud.csv'
    X_train, X_test, y_train, y_test = data()
    model = train_model(X_train, y_train)
    assert isinstance(model, RandomForestClassifier)

def test_save_model():
    """Test saving the model."""
    file_path = 'data/fraud.csv'
    X_train, X_test, y_train, y_test = data()
    model = train_model(X_train, y_train)
    save_model(model, 'fraud_detection', 'random_forest')
