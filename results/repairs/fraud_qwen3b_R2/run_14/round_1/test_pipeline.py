from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
import pytest
from pipeline import load_data, preprocess_data, train_model, predict

def test_load_data():
    data = load_data('data/fraud.csv')
    assert isinstance(data, pd.DataFrame)

def test_preprocess_data():
    data = load_data('data/fraud.csv')
    X_train, y_train = preprocess_data(data)
    assert isinstance(X_train, pd.DataFrame)
    assert isinstance(y_train, pd.Series)

def test_train_model():
    data = load_data('data/fraud.csv')
    X_train, y_train = preprocess_data(data)
    model = train_model(X_train, y_train)
    assert isinstance(model, RandomForestClassifier)

def test_predict():
    data = load_data('data/fraud.csv')
    X_train, y_train = preprocess_data(data)
    model = train_model(X_train, y_train)
    y_pred = predict(model, X_train)
    assert isinstance(y_pred, np.ndarray)
