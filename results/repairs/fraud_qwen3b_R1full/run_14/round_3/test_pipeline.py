import pytest
import pandas as pd
import numpy as np
from pipeline import load_data, preprocess_data, train_model, predict

def test_load_data():
    data = load_data('data/fraud.csv')
    assert isinstance(data, pd.DataFrame)

def test_preprocess_data():
    data = load_data('data/fraud.csv')
    X_train, y_train, X_test, y_test = preprocess_data(data)
    assert isinstance(X_train, pd.DataFrame)
    assert isinstance(y_train, pd.Series)
    assert isinstance(X_test, pd.DataFrame)
    assert isinstance(y_test, pd.Series)

def test_train_model():
    data = load_data('data/fraud.csv')
    X_train, y_train, X_test, y_test = preprocess_data(data)
    model = train_model(X_train, y_train)
    assert isinstance(model, RandomForestClassifier)

def test_predict():
    data = load_data('data/fraud.csv')
    X_train, y_train, X_test, y_test = preprocess_data(data)
    model = train_model(X_train, y_train)
    y_pred = predict(model, X_test)
    assert isinstance(y_pred, np.ndarray)
