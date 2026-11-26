import pytest
from pipeline import load_data, preprocess_data, train_model, predict, evaluate, run_pipeline
import pandas as pd
import numpy as np

def test_load_data():
    """Test loading data from a CSV file."""
    data = load_data('data/fraud.csv')
    assert isinstance(data, pd.DataFrame), "Data should be a pandas DataFrame."

def test_preprocess_data():
    """Test preprocessing the data."""
    data = load_data('data/fraud.csv')
    X, y = preprocess_data(data)
    assert isinstance(X, pd.DataFrame), "X should be a pandas DataFrame."
    assert isinstance(y, pd.Series), "y should be a pandas Series."

def test_train_model():
    """Test training the model."""
    data = load_data('data/fraud.csv')
    X, y = preprocess_data(data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = train_model(X_train, y_train)
    assert isinstance(model, RandomForestClassifier), "Model should be a RandomForestClassifier."

def test_predict():
    """Test making predictions."""
    data = load_data('data/fraud.csv')
    X, y = preprocess_data(data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = train_model(X_train, y_train)
    y_pred = predict(model, X_test)
    assert isinstance(y_pred, np.ndarray), "y_pred should be a numpy array."

def test_evaluate():
    """Test evaluating the model."""
    data = load_data('data/fraud.csv')
    X, y = preprocess_data(data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = train_model(X_train, y_train)
    y_pred = predict(model, X_test)
    accuracy = evaluate(model, X_test, y_test)
    assert isinstance(accuracy, float), "Accuracy should be a float."

def test_run_pipeline():
    """Test running the entire pipeline."""
    accuracy = run_pipeline('data/fraud.csv')
    assert isinstance(accuracy, float), "Accuracy should be a float."
