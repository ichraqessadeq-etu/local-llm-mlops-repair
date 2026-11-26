from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import pytest
from pipeline import load_data, preprocess_data, train_model, evaluate_model

@pytest.fixture
def data():
    return load_data('data/fraud.csv')

def test_load_data(data):
    assert isinstance(data, pd.DataFrame)

def test_preprocess_data(data):
    X_train, y_train, X_test, y_test = preprocess_data(data)
    assert isinstance(X_train, pd.DataFrame)
    assert isinstance(y_train, pd.Series)
    assert isinstance(X_test, pd.DataFrame)
    assert isinstance(y_test, pd.Series)

def test_train_model():
    # Load data
    data = load_data('data/fraud.csv')
    
    # Preprocess data
    X_train, y_train, _, _ = preprocess_data(data)
    
    # Train model
    model = train_model(X_train, y_train)
    
    # Assert that the model is not None
    assert model is not None

def test_evaluate_model():
    # Load data
    data = load_data('data/fraud.csv')
    
    # Preprocess data
    X_train, y_train, X_test, y_test = preprocess_data(data)
    
    # Train model
    model = train_model(X_train, y_train)
    
    # Evaluate model
    evaluate_model(model, X_test, y_test)
